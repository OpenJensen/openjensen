"""Check the smoke probe's lookup against the composed scene, without Isaac."""

import importlib.util
from pathlib import Path
import unittest

from pxr import Usd, UsdPhysics

_ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("_smoke", _ROOT / "smoke_test.py")
_SMOKE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SMOKE)


class _LookupTests(unittest.TestCase):
    def test_lookup_targets_root(self):
        stage = Usd.Stage.Open(str(_ROOT / "scene.usda"))
        roots = {str(prim.GetPath()) for prim in stage.Traverse()
                 if prim.HasAPI(UsdPhysics.ArticulationRootAPI)}

        # Visual descendants must never enter the articulation view's lookup.
        self.assertEqual({_SMOKE._ROBOT_PATTERN}, roots)


if __name__ == "__main__":
    unittest.main()
