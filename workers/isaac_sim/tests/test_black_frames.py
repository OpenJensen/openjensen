import os
from pathlib import Path
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sim_worker import service
from sim_worker.adapters.manifest import load


_DEMO = Path(__file__).resolve().parents[1] / "demo.yaml"
_CUSTOM_SCENE = "/scenes/dark.usda"
_BLACK_RGBA = (0, 0, 0, 255)
_VISIBLE_RGB = (20, 80, 160)


def _pixels(spec, color):
    pixels = MagicMock(ndim=3, shape=(spec.height, spec.width, len(color)), dtype="uint8")

    def select(key):
        # Preserve channel selection so opaque alpha cannot hide black RGB.
        data = bytes(color[key[2]]) * (spec.width * spec.height)
        return SimpleNamespace(tobytes=lambda **kwargs: data)

    pixels.__getitem__.side_effect = select
    return pixels


class BlackFrameTests(unittest.TestCase):
    def _run(self, scene, colors):
        spec = replace(load(_DEMO), scene=scene, width=2, height=2, frames=len(colors))
        names = ("isaacsim", "carb", "carb.settings", "omni", "omni.replicator",
                 "omni.replicator.core", "omni.timeline", "omni.usd", "pxr")
        modules = {name: MagicMock() for name in names}
        for name in names:
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(modules[parent], child, modules[name])

        context = modules["omni.usd"].get_context.return_value
        context.get_stage.return_value.GetTimeCodesPerSecond.return_value = spec.fps
        annotator = modules["omni.replicator.core"].annotators.get.return_value
        annotator.get_data.side_effect = [_pixels(spec, color) for color in colors]

        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {"ACCEPT_EULA": "Y"}),
            patch.object(service.manifest, "load", return_value=spec),
            patch.object(service, "Artifacts", return_value=self.store),
            patch.object(service, "Recorder", return_value=self.recorder),
        ):
            return service.execute(_DEMO, Path("unused"))

    def setUp(self):
        self.store = MagicMock()
        self.store.complete.return_value = {"result_uri": "gs://test/run/result.json"}
        self.recorder = MagicMock()

    def test_black_demo_fails(self):
        with self.assertLogs(level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "entirely black"):
                self._run(load(_DEMO).scene, [_BLACK_RGBA, _BLACK_RGBA])

        self.store.complete.assert_not_called()
        self.store.fail.assert_called_once()
        failure = self.recorder.__exit__.call_args.args[0]
        self.assertIs(failure, RuntimeError)

    def test_visible_demo_passes(self):
        self._run(load(_DEMO).scene, [_BLACK_RGBA, _VISIBLE_RGB])

        self.store.complete.assert_called_once()
        self.store.fail.assert_not_called()

    def test_black_custom_passes(self):
        self._run(_CUSTOM_SCENE, [_BLACK_RGBA, _BLACK_RGBA])

        self.store.complete.assert_called_once()
        self.store.fail.assert_not_called()


if __name__ == "__main__":
    unittest.main()
