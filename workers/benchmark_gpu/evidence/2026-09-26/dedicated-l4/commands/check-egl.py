import json
import os
import pathlib

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import mujoco
from OpenGL import GL

model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 3"/>'
    '<geom type="sphere" size="0.1" rgba="1 0 0 1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, width=64, height=64) as renderer:
    renderer.update_scene(data)
    frame = renderer.render()
    result = {
        "status": "passed",
        "shape": list(frame.shape),
        "renderer": GL.glGetString(GL.GL_RENDERER).decode(),
        "vendor": GL.glGetString(GL.GL_VENDOR).decode(),
        "version": GL.glGetString(GL.GL_VERSION).decode(),
        "nonzero_pixels": int((frame != 0).sum()),
    }
    assert frame.shape == (64, 64, 3) and result["nonzero_pixels"] > 0
    assert "NVIDIA" in result["vendor"] and "L4" in result["renderer"]
(pathlib.Path.home() / "smolvla-benchmark/comparison/egl-preflight.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
