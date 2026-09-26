"""Build the recorded workspace as portable USD; dimensions are estimates."""

import json
import math
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scene.usda"
CONFIG = ROOT / "scene_config.json"
FRAME = ROOT / "evidence/front-frame-000.jpg"
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
GRAVITY = 9.81
CUP_SEGMENTS = 64
COLLIDER_SEGMENTS = 32
STAGE_FPS = 30
STAGE_FRAMES = 150


class Surface(Enum):
    VISUAL = "visual"
    SOLID = "solid"


def _material(stage, name, color, roughness, texture=None):
    path = f"/World/Materials/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if texture is None:
        return material

    reader = UsdShade.Shader.Define(stage, path + "/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    sampler = UsdShade.Shader.Define(stage, path + "/Texture")
    sampler.CreateIdAttr("UsdUVTexture")
    sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"textures/{texture}.png")
    sampler.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    sampler.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    sampler.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.GetInput("diffuseColor").ConnectToSource(sampler.ConnectableAPI(), "rgb")
    return material


def _bind(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _mesh(stage, path, points, faces, material, uv=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(UsdGeom.PointBased(mesh).ComputeExtent(points))
    _bind(mesh.GetPrim(), material)
    if uv is not None:
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        ).Set(uv)
    return mesh


def _cube(stage, path, pos, size, material, surface=Surface.SOLID):
    # Mesh boxes preserve material/UV support across USD renderers.
    points = [(-.5,-.5,-.5),(.5,-.5,-.5),(.5,.5,-.5),(-.5,.5,-.5),
              (-.5,-.5,.5),(.5,-.5,.5),(.5,.5,.5),(-.5,.5,.5)]
    faces = [[3,2,1,0],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[4,5,6,7]]
    cube = _mesh(stage,path,points,faces,material,[(0,0),(1,0),(1,1),(0,1)]*6)
    cube.CreateNormalsAttr([(0,0,-1),(0,-1,0),(1,0,0),(0,1,0),(-1,0,0),(0,0,1)])
    cube.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(cube.GetPrim(), material)
    if surface == Surface.SOLID:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(cube.GetPrim()).CreateApproximationAttr("convexHull")
    return cube


def _quad(stage, path, points, material, repeats=(1, 1)):
    u, v = repeats
    return _mesh(stage, path, points, [[0, 1, 2, 3]], material,
                 [(0, 0), (u, 0), (u, v), (0, v)])


def _textures():
    # Reuse observed surfaces; no generated imagery replaces dataset evidence.
    frame = Image.open(FRAME)
    crops = {
        "wood": (1010, 656, 1480, 725),
        "fabric": (1100, 320, 1228, 448),
        "carpet": (1110, 988, 1500, 1080),
        "cardboard": (630, 794, 836, 862),
        "box_front": (539, 902, 958, 1080),
        "cup_label": (822, 579, 880, 673),
    }
    for name, rect in crops.items():
        frame.crop(rect).save(ROOT / f"textures/{name}.png")
    (ROOT / "evidence/texture_crops.json").write_text(json.dumps(crops, indent=2) + "\n")


def _shelf(stage, cfg, mat):
    shelf = "/World/Environment/Shelf"
    UsdGeom.Xform.Define(stage, shelf)
    width, depth = cfg["width"], cfg["depth"]
    half_w, half_d = width / 2, depth / 2
    beam = cfg["beam_width"]
    for index, height in enumerate(cfg["boards"]):
        p = f"{shelf}/Board{index}"
        _cube(stage, p, (0, 0, height - cfg["board_thickness"] / 2),
              (width, depth, cfg["board_thickness"]), mat["WoodEdge"])
        # The face inherits the cube's scale, so author unit-cube coordinates.
        points = [(-.5, -.5, .5001), (.5, -.5, .5001), (.5, .5, .5001), (-.5, .5, .5001)]
        _quad(stage, p + "/Grain", points, mat["Wood"])
        for side, y in (("Front", -half_d), ("Rear", half_d)):
            _cube(stage, f"{shelf}/Rail{index}{side}", (0, y, height - cfg["board_thickness"] - beam/2),
                  (width, beam, beam), mat["Frame"])

    for i, x in enumerate((-half_w - beam / 2, half_w + beam / 2)):
        for j, y in enumerate((-half_d, half_d)):
            _cube(stage, f"{shelf}/Post{i}{j}", (x, y, (cfg["height"] + beam) / 2),
                  (beam, beam, cfg["height"] - beam), mat["Frame"])
        _cube(stage, f"{shelf}/Foot{i}", (x, 0, beam / 2),
              (beam, depth + beam, beam), mat["Frame"])
    _cube(stage, shelf + "/BottomRail", (0, -half_d, beam / 2),
          (width, beam, beam), mat["Frame"])


def _room(stage, cfg, mat):
    _cube(stage, "/World/Environment/Floor", (0, 0, -.025), (6, 6, .05), mat["CarpetBase"])
    _quad(stage, "/World/Environment/Floor/Weave", [(-.5, -.5, .5001), (.5, -.5, .5001),
          (.5, .5, .5001), (-.5, .5, .5001)], mat["Carpet"], (20, 20))
    for i, x in enumerate((-1.0, 0, 1.0)):
        path = f"/World/Environment/Backdrop/Panel{i}"
        _cube(stage, path, (x, cfg["backdrop_y"], .80), (.99, .065, 1.60), mat["FabricBase"])
        _quad(stage, path + "/Fabric", [(-.5, -.5001, -.5), (.5, -.5001, -.5),
              (.5, -.5001, .5), (-.5, -.5001, .5)], mat["Fabric"], (12, 15))


def _box(stage, cfg, mat):
    path = "/World/Props/Box"
    box = UsdGeom.Xform.Define(stage, path)
    box.AddTranslateOp().Set(Gf.Vec3d(*cfg["position"]))
    w, d, h = cfg["size"]
    t = cfg["thickness"]
    _cube(stage, path + "/Bottom", (0, 0, t / 2), (w, d, t), mat["Cardboard"])
    for side, y in (("Front", -d / 2), ("Rear", d / 2)):
        _cube(stage, path + "/" + side, (0, y, h / 2), (w, t, h), mat["Cardboard"])
    for side, x in (("Left", -w / 2), ("Right", w / 2)):
        _cube(stage, path + "/" + side, (x, 0, h / 2), (t, d, h), mat["Cardboard"])
    _quad(stage, path + "/FrontPrint", [(-w/2, -d/2-t/2-.0001, 0),
          (w/2, -d/2-t/2-.0001, 0), (w/2, -d/2-t/2-.0001, h),
          (-w/2, -d/2-t/2-.0001, h)], mat["BoxPrint"])
    # Thin irregular strips reproduce the visibly torn cardboard rim.
    for side, y in (("Front", -d/2), ("Rear", d/2)):
        xs = np.linspace(-w/2, w/2, 21)
        heights = h + np.random.default_rng(21).uniform(0, .009, len(xs))
        points = [(float(x), y, float(z)) for x,z in zip(xs, heights)]
        points += [(float(x), y, h-.002) for x in xs]
        _mesh(stage, path + "/Torn" + side, points,
              [[i,i+1,i+22,i+21] for i in range(20)], mat["Cardboard"])


def _cup(stage, cfg, mat):
    path = "/World/Props/Cup"
    cup = UsdGeom.Xform.Define(stage, path)
    cup.AddTranslateOp().Set(Gf.Vec3d(*cfg["position"]))
    UsdPhysics.RigidBodyAPI.Apply(cup.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(cup.GetPrim())
    mass.CreateMassAttr(cfg["mass"])
    mass.CreateCenterOfMassAttr((0, 0, cfg["height"] / 2))
    h, r0, r1, t = cfg["height"], cfg["radius_bottom"], cfg["radius_top"], cfg["thickness"]
    radius = (r0+r1)/2
    inertia_xy = cfg["mass"] * (radius**2/2 + h**2/12)
    mass.CreateDiagonalInertiaAttr((inertia_xy, inertia_xy, cfg["mass"]*radius**2))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
    points = []
    for radius, z in ((r0, 0), (r1, h), (r1-t, h), (r0-t, t)):
        points += [(radius*math.cos(i*math.tau/CUP_SEGMENTS),
                    radius*math.sin(i*math.tau/CUP_SEGMENTS), z) for i in range(CUP_SEGMENTS)]
    faces = []
    for band in range(3):
        for i in range(CUP_SEGMENTS):
            j = (i+1)%CUP_SEGMENTS
            faces.append([band*CUP_SEGMENTS+i, band*CUP_SEGMENTS+j,
                          (band+1)*CUP_SEGMENTS+j, (band+1)*CUP_SEGMENTS+i])
    faces.append(list(range(3*CUP_SEGMENTS,4*CUP_SEGMENTS)))
    _mesh(stage, path + "/Paper", points, faces, mat["Paper"])
    # Separate wall hulls preserve the opening; one hull would cap the cup.
    for i in range(COLLIDER_SEGMENTS):
        a, b = i*math.tau/COLLIDER_SEGMENTS, (i+1)*math.tau/COLLIDER_SEGMENTS
        verts = [(r*math.cos(angle),r*math.sin(angle),z)
                 for r,z in ((r0,0),(r1,h),(r1-t,h),(r0-t,0)) for angle in (a,b)]
        hull = _mesh(stage, f"{path}/Collision/Wall{i:02}", verts,
                     [[0,1,3,2],[2,3,5,4],[4,5,7,6],[6,7,1,0],[0,2,4,6],[1,7,5,3]], mat["Paper"])
        hull.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(hull.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(hull.GetPrim()).CreateApproximationAttr("convexHull")
    bottom = UsdGeom.Cylinder.Define(stage, path + "/Collision/Bottom")
    bottom.CreateRadiusAttr(r0)
    bottom.CreateHeightAttr(t)
    bottom.AddTranslateOp().Set((0,0,t/2))
    bottom.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(bottom.GetPrim())
    # Front label follows the taper, retaining the photographed printing.
    label_points, label_faces, label_uv = [], [], []
    count = 20
    for i in range(count+1):
        angle = -math.pi/2 + (i/count-.5)*.85
        for z in (.012,h-.018):
            radius = r0 + (r1-r0)*z/h + .00015
            label_points.append((radius*math.cos(angle), radius*math.sin(angle),z))
    for i in range(count):
        label_faces.append([2*i,2*i+2,2*i+3,2*i+1])
        label_uv += [(i/count,0),((i+1)/count,0),((i+1)/count,1),(i/count,1)]
    _mesh(stage, path + "/Label", label_points, label_faces, mat["CupPrint"], label_uv)
    contact = UsdShade.Material.Define(stage, "/World/Materials/CupContact")
    physical = UsdPhysics.MaterialAPI.Apply(contact.GetPrim())
    physical.CreateStaticFrictionAttr(.65)
    physical.CreateDynamicFrictionAttr(.5)
    physical.CreateRestitutionAttr(.05)
    UsdShade.MaterialBindingAPI.Apply(cup.GetPrim()).Bind(contact, materialPurpose="physics")


def _cables(stage, mat):
    # Cables are visual details; they do not obstruct manipulation physics.
    for name, color in (("USB", "Cable"), ("Power", "Paper")):
        curve = UsdGeom.BasisCurves.Define(stage, "/World/Environment/" + name)
        points = [(-.43,.02,.26),(-.49,.00,.23),(-.55,-.10,.11),
                  (-.55,-.30,.014),(-.62,-.55,.006),(-.79,-.72,.006)]
        if name == "Power":
            points = [(x-.02,y+.07,z) for x,y,z in points]
        curve.CreatePointsAttr(points)
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreateTypeAttr("linear")
        curve.CreateWidthsAttr([.003])
        curve.SetWidthsInterpolation("constant")
        _bind(curve.GetPrim(), mat[color])


def _camera(stage, name, position, target, focal, roll_degrees=0):
    camera = UsdGeom.Camera.Define(stage, "/World/Cameras/" + name)
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(0,0,1)).GetInverse()
    matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),roll_degrees)) * matrix
    camera.AddTransformOp().Set(matrix)
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateVerticalApertureAttr(36*CAMERA_HEIGHT/CAMERA_WIDTH)
    camera.CreateClippingRangeAttr((.01,100))
    return camera


def _lighting(stage, cfg):
    fill = UsdLux.DomeLight.Define(stage, "/World/Lights/Fill")
    fill.CreateIntensityAttr(cfg["fill_intensity"])
    fill.CreateColorAttr((.93,.96,1))
    key = UsdLux.RectLight.Define(stage, "/World/Lights/Window")
    key.CreateIntensityAttr(cfg["key_intensity"])
    key.CreateWidthAttr(1.4)
    key.CreateHeightAttr(1.8)
    key.CreateColorAttr((1,.95,.87))
    key.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(1,-1.1,1.9),Gf.Vec3d(0,0,.25),Gf.Vec3d(0,0,1)).GetInverse())


def _build():
    cfg = json.loads(CONFIG.read_text())
    _textures()
    stage = Usd.Stage.CreateNew(str(SCENE))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetTimeCodesPerSecond(STAGE_FPS)
    stage.SetFramesPerSecond(STAGE_FPS)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(STAGE_FRAMES)
    stage.GetRootLayer().customLayerData = {"dataset": cfg["dataset"], "fidelity": "visual reconstruction; unmeasured dimensions"}
    physics = UsdPhysics.Scene.Define(stage, "/World/Physics")
    physics.CreateGravityDirectionAttr((0,0,-1))
    physics.CreateGravityMagnitudeAttr(GRAVITY)
    physics.GetPrim().AddAppliedSchema("PhysxSceneAPI")
    physics.GetPrim().CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(120)
    mat = {name:_material(stage,name,tuple(v["color"]),v["roughness"],v.get("texture"))
           for name,v in cfg["materials"].items()}
    _room(stage,cfg["room"],mat)
    _shelf(stage,cfg["shelf"],mat)
    _box(stage,cfg["box"],mat)
    _cup(stage,cfg["cup"],mat)
    _cables(stage,mat)

    robot = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot.GetPrim().GetReferences().AddReference("robot/so101.usda")
    robot.AddTranslateOp().Set(Gf.Vec3d(*cfg["robot"]["position"]))
    robot.AddRotateZOp().Set(cfg["robot"]["yaw_degrees"])
    _camera(stage,"Front",**cfg["camera"])
    _camera(stage,"Overview",(1.35,-1.8,1.2),(0,-.05,.40),40)
    _lighting(stage,cfg["lighting"])
    stage.GetRootLayer().Save()
    print(SCENE)


if __name__ == "__main__":
    _build()
