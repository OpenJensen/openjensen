"""Render the USD geometry for visual review; this does not test Isaac physics."""

import sys
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parent
RESOLUTION = (1920, 1080)
SAMPLES = 32
PREVIEW_FILL = 0.35
PREVIEW_KEY_WATTS = 18


def _render():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(ROOT / "scene.usda"), import_cameras=True,
                          import_lights=True, import_materials=True, import_guide=False)
    scene = bpy.context.scene
    # Blender and Isaac interpret imported light intensity differently.
    if scene.world and scene.world.use_nodes:
        for node in scene.world.node_tree.nodes:
            if node.type == "BACKGROUND":
                node.inputs["Strength"].default_value = PREVIEW_FILL
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.data.energy = PREVIEW_KEY_WATTS
    scene.camera = next(obj for obj in scene.objects if obj.type == "CAMERA" and obj.name == "Front")
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(ROOT / "preview.png")
    if "--" in sys.argv:
        scene.render.filepath = sys.argv[sys.argv.index("--")+1]
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    _render()
