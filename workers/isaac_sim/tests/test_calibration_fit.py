import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim_worker.calibration_fit import CalibrationFit

_SCENE = Path(__file__).parents[1] / "scenes/so101-pickup"
_URDF = _SCENE / "robot/source/so101_new_calib.urdf"
_SIGNS = (1, -1, 1, 1, -1)
_LANDMARKS = {
    "shoulder": {"link": "shoulder_link", "xyz": [0.01, 0.02, 0.03]},
    "upper": {"link": "upper_arm_link", "xyz": [0.01, 0.02, 0.03]},
    "lower": {"link": "lower_arm_link", "xyz": [0.01, 0.02, 0.03]},
    "wrist": {"link": "wrist_link", "xyz": [0.01, 0.02, 0.03]},
    "fixed_jaw": {"link": "gripper_link", "xyz": [0.02, 0.03, -0.08]},
    "moving_jaw": {"link": "moving_jaw_so101_v1_link", "xyz": [0.01, -0.07, 0.03]},
}
_TRUTH = np.array([0.2, -0.1, 0.1, 0.02, 0.04, 1.2, 800, 0.1, -0.06, 0.2, -0.1, 0.03, 0.012])
_LOWER = [-1, -1, -1, -1, -1, 0.5, 200, -0.5, -0.5, -0.5, -0.5, -0.5, 0.001]
_UPPER = [1, 1, 1, 1, 1, 3, 2000, 0.5, 0.5, 0.5, 0.5, 0.5, 0.03]


class CalibrationFitTests(unittest.TestCase):
    def setUp(self):
        self.fitter = CalibrationFit(_URDF, _LANDMARKS, (640, 360), _SIGNS)

    def _rows(self, count, seed):
        random = np.random.default_rng(seed)
        rows = []
        for _ in range(count):
            state = random.uniform([-40, -70, -70, -60, -100, 0], [40, 70, 70, 60, 100, 90])
            points = self.fitter.project(state, _TRUTH)
            rows.append({"state": state.tolist(), "points": points})
        return rows

    def test_recovers_mapping(self):
        initial = _TRUTH + [0.02, -0.03, 0.04, 0.01, -0.02, 0.06, 60,
                            -0.03, 0.02, -0.03, 0.04, -0.02, 0.001]
        result = self.fitter.fit(self._rows(15, 1), self._rows(5, 2), initial, _LOWER, _UPPER)
        np.testing.assert_allclose(result["parameters"], _TRUTH, atol=1e-7)
        self.assertLess(result["validation"]["rms_px"], 1e-7)
        self.assertEqual(result["uncertainty"]["jacobian_rank"], len(_TRUTH))
        self.assertEqual(result["verification"], "unverified")

    def test_holdout_stays_separate(self):
        validation = self._rows(4, 3)
        for row in validation:
            for point in row["points"].values():
                point[0] += 30
                point[1] += 40
        result = self.fitter.fit(self._rows(12, 4), validation, _TRUTH, _LOWER, _UPPER)
        np.testing.assert_allclose(result["parameters"], _TRUTH, atol=1e-8)
        self.assertAlmostEqual(result["validation"]["rms_px"], 50)

    def test_unobservable_gripper(self):
        rows = self._rows(12, 5)
        for row in rows:
            del row["points"]["moving_jaw"]
        result = self.fitter.fit(rows, [], _TRUTH, _LOWER, _UPPER)
        self.assertLess(result["uncertainty"]["jacobian_rank"], len(_TRUTH))
        self.assertIsNone(result["uncertainty"]["standard_errors"])
        self.assertIsNone(result["validation"])

    def test_non_z_axis_and_tool(self):
        joints = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll", "gripper"]
        links = [f"<link name='link{index}'/>" for index in range(len(joints) + 1)]
        chain = [
            f"<joint name='{name}' type='revolute'><parent link='link{index}'/>"
            f"<child link='link{index + 1}'/><axis xyz='1 0 0'/></joint>"
            for index, name in enumerate(joints)
        ]
        chain.append("<joint name='tool' type='fixed'><parent link='link6'/>"
                     "<child link='tip'/><origin xyz='0 1 0' rpy='0 0 1.5707963267948966'/>"
                     "</joint><link name='tip'/>")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.urdf"
            path.write_text("<robot>" + "".join(links + chain) + "</robot>")
            fitter = CalibrationFit(path, {"tip": {"link": "tip", "xyz": [1, 0, 0]}},
                                    (640, 360), (1, 1, 1, 1, 1))
            parameters = [0, 0, 0, 0, 0, 4, 800, 0, 0, 0, 0, 0, 0.01]
            # Rx(90°) maps tool (0,2,0) to (0,0,2), then camera adds depth4.
            pixel = fitter.project([90, 0, 0, 0, 0, 0], parameters)["tip"]
            np.testing.assert_allclose(pixel, [320, 180], atol=1e-9)
            parameters[7] = -math.pi / 2
            pixel = fitter.project([90, 0, 0, 0, 0, 0], parameters)["tip"]
            np.testing.assert_allclose(pixel, [320, 580], atol=1e-9)

    def test_invalid_projection(self):
        parameters = _TRUTH.copy()
        parameters[5] = -3
        with self.assertRaisesRegex(ValueError, "front"):
            self.fitter.project([0] * 6, parameters)
        with self.assertRaisesRegex(ValueError, "signs"):
            CalibrationFit(_URDF, _LANDMARKS, (640, 360), (1, 1, 0, 1, 1))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.fitter.project([math.nan] * 6, _TRUTH)


if __name__ == "__main__":
    unittest.main()
