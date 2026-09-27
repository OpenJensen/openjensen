"""Deterministic ground-truth trajectories; these are not robot performance tests."""

import math
import tempfile
import unittest
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path

from sim_worker.rollout.contracts import Frame, ObjectState, Observation
from sim_worker.rollout.evaluation import EpisodeEvaluator, EvaluationSpec, world_bounds


def criteria(task="lift_hold"):
    return EvaluationSpec.parse(
        {
            "task": task,
            "object_prim": "/World/Props/Cup",
            "support_height_m": 0.2,
            "lift_clearance_m": 0.01,
            "lift_hold_seconds": 0.2,
            "settle_seconds": 0.2,
            "maximum_linear_speed_m_s": 0.02,
            "maximum_angular_speed_rad_s": 0.1,
            "placement_min_m": [0.8, -0.2, 0.0] if task == "lift_place" else None,
            "placement_max_m": [1.2, 0.2, 0.2] if task == "lift_place" else None,
        }
    )


def sample(step, *, xyz=None, velocity=(0, 0, 0), angular=(0, 0, 0), quaternion=(0, 0, 0, 1)):
    return Observation(
        "episode",
        step,
        step / 10,
        (0,),
        Frame(2, 2, bytes(12)),
        ObjectState(
            "/World/Props/Cup",
            xyz or (0, 0, 0.2 if step == 0 else 0.22),
            quaternion,
            velocity,
            angular,
            (-0.04, -0.04, 0),
            (0.04, 0.04, 0.1),
        ),
    )


def score(samples, task="lift_hold"):
    evaluator = EpisodeEvaluator(criteria(task), fps=10, steps=len(samples) - 1)
    for observation in samples:
        evaluator.observe(observation)
    return evaluator.finish()


def cases(names, values):
    def decorate(function):
        @wraps(function)
        def run(self):
            for value in values:
                args = value if "," in names else (value,)
                with self.subTest(**dict(zip(names.split(","), args))):
                    function(self, *args)

        return run

    return decorate


class EvaluationTests(unittest.TestCase):
    def test_whole_body_lift_then_stable_terminal_hold(self):
        result = score([sample(i) for i in range(6)])
        assert result["geometric_success"] is True
        assert result["samples"] == 6
        assert result["criteria_sha256"] == criteria().digest
        assert result["quality_validated"] is False
        assert result["grasp_and_release_verified"] is False
        self.assertAlmostEqual(result["final"]["terminal_settle_seconds"], 0.2)

    def test_success_is_not_latched_after_object_drops(self):
        observations = [sample(i) for i in range(7)]
        observations[-1] = sample(6, xyz=(0, 0, 0.2))
        result = score(observations)
        assert result["final"]["lift_met"] is True
        assert result["geometric_success"] is False

    def test_short_second_lift_cannot_reuse_previous_hold(self):
        spec = replace(criteria(), settle_seconds=0.1)
        evaluator = EpisodeEvaluator(spec, fps=10, steps=6)
        for i, height in enumerate((0.2, 0.22, 0.22, 0.22, 0.2, 0.22, 0.22)):
            evaluator.observe(sample(i, xyz=(0, 0, height)))
        result = evaluator.finish()
        self.assertTrue(result["final"]["lift_met"])
        self.assertFalse(result["final"]["current_lift_hold_met"])
        self.assertFalse(result["geometric_success"])

    def test_full_placement_requires_prior_lift_and_terminal_rest(self):
        samples = [sample(i) for i in range(4)] + [
            sample(i, xyz=(1, 0, 0.003)) for i in range(4, 8)
        ]
        assert score(samples, "lift_place")["geometric_success"] is True
        no_lift = [sample(0)] + [sample(i, xyz=(1, 0, 0.003)) for i in range(1, 8)]
        assert score(no_lift, "lift_place")["geometric_success"] is False
        samples[-1] = sample(7, xyz=(1, 0, 0.003), velocity=(0.5, 0, 0))
        assert score(samples, "lift_place")["geometric_success"] is False

    def test_center_in_box_is_insufficient(self):
        samples = [sample(i) for i in range(4)] + [
            sample(i, xyz=(1.19, 0, 0.003)) for i in range(4, 8)
        ]
        result = score(samples, "lift_place")
        assert result["final"]["inside_placement_region"] is False
        assert result["geometric_success"] is False

    def test_tipped_cup_on_shelf_is_not_a_lift(self):
        tilted = (0, math.sqrt(0.5), 0, math.sqrt(0.5))
        # Its origin is above the shelf, but its full geometry still touches it.
        samples = [sample(0)] + [
            sample(i, xyz=(0, 0, 0.24), quaternion=tilted) for i in range(1, 8)
        ]
        assert score(samples)["geometric_success"] is False

    @cases(
        "change",
        [
            {"velocity": (0.1, 0, 0)},
            {"angular": (0, 0.2, 0)},
        ],
    )
    def test_motion_breaks_terminal_settle(self, change):
        samples = [sample(i) for i in range(7)]
        samples[5] = sample(5, **change)
        assert score(samples)["geometric_success"] is False

    def test_discontinuous_lift_cannot_accumulate_hold_time(self):
        samples = [sample(i, xyz=(0, 0, 0.2 if i % 2 == 0 else 0.22)) for i in range(8)]
        assert score(samples)["final"]["lift_met"] is False

    @cases(
        "change",
        [
            {"episode_id": "other"},
            {"step": 2},
            {"step": True},
            {"sim_time": 0.2},
            {"sim_time": math.nan},
            {"object_state": None},
        ],
    )
    def test_missing_stale_or_misaligned_observations_rejected(self, change):
        evaluator = EpisodeEvaluator(criteria(), fps=10, steps=5)
        evaluator.observe(sample(0))
        with self.assertRaises(ValueError):
            evaluator.observe(replace(sample(1), **change))

    @cases(
        "change",
        [
            {"prim_path": "/World/Other"},
            {"orientation_xyzw": (1, 1, 1, 1)},
            {"position_m": (0, math.inf, 0)},
            {"linear_velocity_m_s": (0, math.nan, 0)},
            {"local_max_m": (0.05, 0.04, 0.1)},
        ],
    )
    def test_mutated_or_invalid_object_rejected(self, change):
        evaluator = EpisodeEvaluator(criteria(), fps=10, steps=5)
        evaluator.observe(sample(0))
        observation = sample(1)
        with self.assertRaises(ValueError):
            evaluator.observe(
                replace(observation, object_state=replace(observation.object_state, **change))
            )

    @cases("xyz,task", [((0, 0, 0.3), "lift_hold"), ((1, 0, 0.003), "lift_place")])
    def test_initial_success_rejected_before_motion(self, xyz, task):
        evaluator = EpisodeEvaluator(criteria(task), fps=10, steps=5)
        with self.assertRaisesRegex(ValueError, "already"):
            evaluator.observe(sample(0, xyz=xyz))

    def test_incomplete_episode_never_reported_as_failure_or_success(self):
        evaluator = EpisodeEvaluator(criteria(), fps=10, steps=5)
        evaluator.observe(sample(0))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            evaluator.finish()

    @cases(
        "field,value",
        [
            ("task", "grasp"),
            ("task", []),
            ("object_prim", "/World/*"),
            ("lift_clearance_m", False),
            ("lift_hold_seconds", 0),
            ("settle_seconds", math.nan),
            ("placement_min_m", [0, 0, 0]),
            ("maximum_linear_speed_m_s", -1),
        ],
    )
    def test_invalid_criteria_rejected(self, field, value):
        document = asdict(criteria()) | {field: value}
        with self.assertRaises(ValueError):
            EvaluationSpec.parse(document)

    def test_explicit_criteria_identity_cannot_ignore_threshold_change(self):
        spec = criteria()
        assert spec.digest != replace(spec, settle_seconds=0.3).digest
        with self.assertRaises(ValueError):
            EvaluationSpec.parse(asdict(spec) | {"pretend_success": True})

    def test_quaternion_order_and_rotation_include_all_bounds(self):
        body = sample(
            0, xyz=(1, 2, 3), quaternion=(0, math.sqrt(0.5), 0, math.sqrt(0.5))
        ).object_state
        lower, upper = world_bounds(body)
        for actual, expected in zip(lower, (1, 1.96, 2.96)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(upper, (1.1, 2.04, 3.04)):
            self.assertAlmostEqual(actual, expected)

    def test_rollout_writes_replayable_object_evidence_without_claiming_quality(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tmp_path = Path(directory.name)
        from test_rollout import _calibration, _Policy, _Simulation, _spec

        from sim_worker.rollout.calibration import JointMap
        from sim_worker.rollout.service import Rollout

        class MeasuredSimulation(_Simulation):
            def observe(self):
                observation = super().observe()
                return replace(observation, object_state=sample(self.index).object_state)

        calibration = tmp_path / "calibration.yaml"
        _calibration(calibration)
        spec = replace(_spec(calibration), steps=16, evaluation=criteria())
        sim = MeasuredSimulation()
        records = []
        result = Rollout(spec, sim, _Policy(sim), JointMap(calibration, spec.sim.joints)).run(
            "episode", lambda values, _: records.append(values)
        )
        assert result["evaluation"]["geometric_success"] is True
        assert result["evaluation"]["samples"] == 17
        assert result["evaluation"]["quality_validated"] is False
        assert len(records) == 16
        assert records[0]["object_state"]["position_m"][2] == 0.2
        assert records[-1]["next_object_state"]["position_m"][2] == 0.22
        assert records[-1]["evaluation"]["step"] == 16
        from sim_worker.rollout.evaluate_trace import evaluate

        self.assertEqual(evaluate(spec, records), result["evaluation"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            evaluate(spec, records[:-1])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            evaluate(spec, records[:4] + records[5:])
        changed = [dict(row) for row in records]
        changed[2]["object_state"] = dict(changed[2]["object_state"], position_m=(0, 0, 0.23))
        with self.assertRaisesRegex(ValueError, "discontinuous"):
            evaluate(spec, changed)
