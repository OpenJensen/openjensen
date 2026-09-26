import copy
import math
import unittest

from sim_worker.calibration_fit.validation import evaluate

_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_NAMES = [
    "camera_rx",
    "camera_ry",
    "camera_rz",
    "camera_tx",
    "camera_ty",
    "camera_tz",
    "focal_px",
    "shoulder_lift_offset",
    "elbow_flex_offset",
    "wrist_flex_offset",
    "wrist_roll_offset",
    "gripper_offset",
    "gripper_slope",
]


class CalibrationValidationTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "parameter_names": _NAMES.copy(),
            "parameters": [0, 0, 0, 0, 0, 1, 800, 0, 0, 0, 0, 0, 0.01],
            "signs": [1, -1, 1, 1, 1],
            "pan_offset_rad": 0,
            "solver": {"success": True, "active_bounds": [0] * len(_NAMES)},
            "fit": {"frames": 8, "points": 32, "behind_camera_points": 0},
            "validation": {
                "frames": 4,
                "points": 16,
                "behind_camera_points": 0,
                "rms_px": 5,
                "max_px": 10,
            },
            "uncertainty": {
                "jacobian_rank": len(_NAMES),
                "standard_errors": [0.0001] * len(_NAMES),
            },
            "verification": "unverified",
        }
        self.support = {
            source: {"min": [-30, -20, -15, -10, -5, 0], "max": [30, 20, 15, 10, 5, 40]}
            for source in ("state", "action")
        }
        self.limits = {name: [-2, 2] for name in _JOINTS}
        self.metadata = {
            "fit_episodes": [1, 2],
            "validation_episodes": [3],
            "pan_offset_gauge": "fixed_zero",
            "landmark_geometry": {
                "status": "proven",
                "source": "synthetic known geometry",
                "independent": True,
                "uncertainty_included": True,
            },
            "sign_ambiguity": {"status": "resolved", "source": "synthetic known signs"},
        }

    def _evaluate(self):
        return evaluate(self.report, self.support, self.limits, self.metadata)

    def test_complete_evidence(self):
        original = copy.deepcopy(self.report)
        result = self._evaluate()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["failures"], [])
        self.assertEqual(self.report, original)
        self.assertAlmostEqual(result["mapped_support_rad"]["action"]["gripper"][1], 0.4)

    def test_metadata_is_not_inferred(self):
        result = evaluate(self.report, self.support, self.limits, {})
        self.assertEqual(result["status"], "unverified")
        for check in ("episodes", "pan_gauge", "geometry", "sign_provenance"):
            self.assertFalse(result["checks"][check])

    def test_disjoint_episodes(self):
        self.metadata["validation_episodes"] = ["1"]
        self.assertFalse(self._evaluate()["checks"]["episodes"])

    def test_solver_bounds_depth(self):
        self.report["solver"]["success"] = False
        self.report["solver"]["active_bounds"][7] = 1
        self.report["fit"]["behind_camera_points"] = 1
        self.report["validation"]["behind_camera_points"] = 1
        checks = self._evaluate()["checks"]
        for name in ("solver", "bounds", "fit_depth", "validation_depth"):
            self.assertFalse(checks[name])

    def test_heldout_errors(self):
        for metric, value in (("rms_px", 9.01), ("max_px", 24.01), ("rms_px", math.nan)):
            with self.subTest(metric=metric, value=value):
                self.report["validation"][metric] = value
                self.assertFalse(self._evaluate()["checks"][metric])

    def test_action_support(self):
        self.support["action"]["max"][1] = 130
        checks = self._evaluate()["checks"]
        self.assertTrue(checks["state.shoulder_lift"])
        self.assertFalse(checks["action.shoulder_lift"])

    def test_gripper_full_range(self):
        self.report["uncertainty"]["standard_errors"][-1] = 0.001
        result = self._evaluate()
        self.assertFalse(result["checks"]["gripper_uncertainty"])
        self.assertAlmostEqual(result["gripper_standard_error_bound_degrees"], math.degrees(0.0401))

    def _add_covariance(self):
        errors = self.report["uncertainty"]["standard_errors"]
        errors[-2:] = [0.02, 0.001]
        matrix = [[0.0] * len(errors) for _ in errors]
        for index, error in enumerate(errors):
            matrix[index][index] = error**2
        matrix[-2][-1] = matrix[-1][-2] = -0.00002
        self.report["uncertainty"]["covariance"] = matrix
        return matrix

    def test_correlated_gripper(self):
        self._add_covariance()
        result = self._evaluate()
        self.assertTrue(result["checks"]["gripper_uncertainty"])
        self.assertAlmostEqual(result["gripper_standard_error_bound_degrees"], math.degrees(0.02))
        self.assertEqual(result["gripper_uncertainty_method"], "propagated_covariance")

    def test_covariance_action_range(self):
        self._add_covariance()
        self.support["action"]["max"][-1] = 60
        result = self._evaluate()
        self.assertFalse(result["checks"]["gripper_uncertainty"])
        self.assertAlmostEqual(result["gripper_standard_error_bound_degrees"], math.degrees(0.04))

    def test_invalid_covariance(self):
        for case in ("asymmetric", "indefinite", "nonfinite", "diagonal_mismatch", "shape"):
            with self.subTest(case=case):
                matrix = self._add_covariance()
                if case == "asymmetric":
                    matrix[-2][-1] = 0
                if case == "indefinite":
                    # An invalid arm block must fail even when the gripper block is valid.
                    matrix[7][8] = matrix[8][7] = 0.0001
                if case == "nonfinite":
                    matrix[0][0] = math.nan
                if case == "diagonal_mismatch":
                    matrix[7][7] *= 4
                if case == "shape":
                    matrix.pop()
                result = self._evaluate()
                self.assertFalse(result["checks"]["covariance"])
                self.assertFalse(result["checks"]["gripper_uncertainty"])
                self.assertIsNone(result["gripper_standard_error_bound_degrees"])

    def test_missing_uncertainty(self):
        self.report["uncertainty"]["jacobian_rank"] -= 1
        self.report["uncertainty"]["standard_errors"] = None
        checks = self._evaluate()["checks"]
        self.assertFalse(checks["rank"])
        self.assertFalse(checks["standard_errors"])
        self.assertFalse(checks["gripper_uncertainty"])

    def test_frame_lag_rank(self):
        self.report["parameter_names"].append("frame_lag")
        self.report["parameters"].append(-2.5)
        self.report["solver"]["active_bounds"].append(0)
        self.report["uncertainty"]["standard_errors"].append(0.02)
        self.assertFalse(self._evaluate()["checks"]["rank"])
        self.report["uncertainty"]["jacobian_rank"] += 1
        self.assertEqual(self._evaluate()["status"], "verified")

    def test_parameter_order(self):
        for name in ("parameter_names", "parameters"):
            self.report[name].reverse()
        self.report["uncertainty"]["standard_errors"].reverse()
        self.assertEqual(self._evaluate()["status"], "verified")

    def test_missing_support_limits(self):
        self.support["action"]["max"].pop()
        del self.limits["gripper"]
        checks = self._evaluate()["checks"]
        self.assertFalse(checks["support"])
        self.assertFalse(checks["limits"])

    def test_geometry_uncertainty(self):
        self.metadata["landmark_geometry"]["uncertainty_included"] = False
        self.assertFalse(self._evaluate()["checks"]["geometry"])

    def test_offset_standard_error(self):
        index = self.report["parameter_names"].index("wrist_flex_offset")
        self.report["uncertainty"]["standard_errors"][index] = math.radians(2)
        self.assertTrue(self._evaluate()["checks"]["wrist_flex_offset"])
        self.report["uncertainty"]["standard_errors"][index] = math.radians(2.01)
        self.assertFalse(self._evaluate()["checks"]["wrist_flex_offset"])


if __name__ == "__main__":
    unittest.main()
