import os
from pathlib import Path
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sim_worker.adapters import isaac
from sim_worker.adapters.manifest import load


_DEMO = Path(__file__).resolve().parents[1] / "demo.yaml"
_PATH_TRACING = "PathTracing"
_AA_DISABLED = 0
_SAMPLES_PER_PIXEL = 64
_DENOISER_BLEND = "/rtx/pathtracing/optixDenoiser/blendFactor"


class RenderTests(unittest.TestCase):
    def test_recording_avoids_dlss(self):
        app = MagicMock()
        sdk = SimpleNamespace(SimulationApp=MagicMock(return_value=app))
        with (
            patch.dict(sys.modules, {"isaacsim": sdk}),
            patch.dict(os.environ, {"ACCEPT_EULA": "Y"}),
            patch.object(isaac, "_capture", return_value=(frame for frame in ())),
        ):
            with isaac.frames(load(_DEMO)) as capture:
                list(capture)

        # Offline capture needs its own sampling and denoising, without NGX.
        config = sdk.SimulationApp.call_args.args[0]
        self.assertEqual(config.get("renderer"), _PATH_TRACING)
        self.assertEqual(config.get("anti_aliasing"), _AA_DISABLED)
        self.assertEqual(config.get("samples_per_pixel_per_frame"), _SAMPLES_PER_PIXEL)
        self.assertIs(config.get("denoiser"), True)

    def test_stage_keeps_quality(self):
        names = ("isaacsim", "carb", "carb.settings", "omni", "omni.replicator",
                 "omni.replicator.core", "omni.timeline", "omni.usd", "pxr")
        modules = {name: MagicMock() for name in names}
        for name in names:
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(modules[parent], child, modules[name])

        spec = replace(load(_DEMO), width=2, height=2, frames=3)
        app = modules["isaacsim"].SimulationApp.return_value
        context = modules["omni.usd"].get_context.return_value
        context.get_stage.return_value.GetTimeCodesPerSecond.return_value = spec.fps
        rep = modules["omni.replicator.core"]
        events = []
        context.open_stage.side_effect = lambda scene: events.append("opened") or True
        app.reset_render_settings.side_effect = lambda: events.append("configured")
        rep.orchestrator.step.side_effect = lambda **kwargs: events.append("rendered")
        pixels = MagicMock(ndim=3, shape=(spec.height, spec.width, 4), dtype="uint8")
        pixels.__getitem__.return_value.tobytes.return_value = b"\xff" * spec.frame_bytes
        rep.annotators.get.return_value.get_data.return_value = pixels

        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"ACCEPT_EULA": "Y"}):
            with isaac.frames(spec) as capture:
                recorded = list(capture)

        # Stage loading must not replace the recording profile before capture.
        self.assertEqual(events[:3], ["opened", "configured", "rendered"])
        settings = modules["carb.settings"].get_settings.return_value
        settings.set.assert_any_call(_DENOISER_BLEND, 0.0)
        self.assertEqual(len(recorded), spec.frames)
        steps = [call.kwargs for call in rep.orchestrator.step.call_args_list]
        self.assertEqual(len(steps), spec.frames + 1)
        self.assertEqual(steps[0]["delta_time"], 0.0)
        for step in steps[1:]:
            self.assertEqual(step["delta_time"], 1.0 / spec.fps)
            self.assertGreater(step.get("rt_subframes", 0), 1)
            self.assertIs(step.get("wait_for_render", True), True)


if __name__ == "__main__":
    unittest.main()
