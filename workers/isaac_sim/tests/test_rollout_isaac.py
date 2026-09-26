import os
import sys
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sim_worker.rollout.contracts import SimSpec
from sim_worker.rollout.isaac import IsaacSim

_SPEC = SimSpec("scene.usda", "/World/Camera", "/World/Robot/root", ("pan", "elbow"), 2, 2, 30, 60)


class _World:
    def __init__(self):
        self.time = 0.0
        self.robot = MagicMock()
        self.robot.count = 1
        self.robot.max_dofs = 2
        self.robot.shared_metatype.dof_names = ("elbow", "pan")
        self.robot.get_dof_limits.return_value = np.array([[[-1.0, 1.0], [-2.0, 2.0]]])
        self.robot.get_dof_positions.side_effect = lambda: self.positions.copy()
        self.robot.get_dof_position_targets.side_effect = lambda: self.targets.copy()
        self.robot.set_dof_position_targets.side_effect = self._apply
        self.view = MagicMock()
        self.view.create_articulation_view.return_value = self.robot
        self.manager = MagicMock()
        self.manager.get_physics_simulation_view.return_value = self.view
        self.manager.get_simulation_time.side_effect = lambda: self.time
        self.manager.is_fabric_enabled.return_value = True
        self.manager.step.side_effect = self._step
        self.stage = MagicMock()
        self.stage.Reload.side_effect = self._reload
        self.context = MagicMock()
        self.context.open_stage.return_value = True
        self.context.get_stage.return_value = self.stage
        self.rep = MagicMock()
        self.rep.annotators.get.return_value.get_data.return_value = np.full(
            (2, 2, 4), [10, 20, 30, 255], dtype=np.uint8
        )
        self.timeline = MagicMock()
        self._reload()

    def _reload(self):
        self.time = 0.0
        self.positions = np.array([[0.25, 0.75]], dtype=np.float32)
        self.targets = self.positions.copy()
        self.cup_position = 0.0
        self.cup_velocity = 0.0

    def _apply(self, values, indices):
        self.targets = values.copy()

    def _step(self, steps, update_fabric):
        self.time += steps / _SPEC.physics_hz
        self.positions = self.targets.copy()
        self.cup_position += 1.0
        self.cup_velocity = 1.0

    def _load(self, driver):
        driver._rep = self.rep
        driver._context = self.context
        driver._timeline = self.timeline
        driver._manager = self.manager
        driver._geom = SimpleNamespace(Camera=object())


class IsaacDriverTests(unittest.TestCase):
    def setUp(self):
        self.world = _World()
        self.app = MagicMock()
        sdk = SimpleNamespace(SimulationApp=MagicMock(return_value=self.app))
        self.enterContext(patch.dict(sys.modules, {"isaacsim": sdk}))
        self.enterContext(patch.dict(os.environ, {"ACCEPT_EULA": "Y"}))
        self.enterContext(
            patch.object(IsaacSim, "_load_sdk", lambda driver: self.world._load(driver))
        )
        self.driver = self.enterContext(IsaacSim(_SPEC))

    def test_joint_order_and_rgb(self):
        self.driver.reset("episode-1")
        observed = self.driver.observe()
        self.assertEqual(observed.state, (0.75, 0.25))
        self.assertEqual(observed.frame.rgb, bytes([10, 20, 30]) * 4)
        self.assertEqual(
            (observed.episode_id, observed.step, observed.sim_time), ("episode-1", 0, 0.0)
        )

    def test_limits_in_policy_order(self):
        with self.assertRaisesRegex(RuntimeError, "Reset"):
            _ = self.driver.joint_limits
        self.driver.reset("episode-1")
        self.assertEqual(self.driver.joint_limits, ((-2.0, 2.0), (-1.0, 1.0)))

    def test_observe_keeps_clock(self):
        self.driver.reset("episode-1")
        for _ in range(3):
            self.assertEqual(self.driver.observe().sim_time, 0.0)
        self.world.manager.step.assert_not_called()
        for call in self.world.rep.orchestrator.step.call_args_list:
            self.assertEqual(call.kwargs["delta_time"], 0.0)
            self.assertTrue(call.kwargs["wait_for_render"])

    def test_targets_and_exact_step(self):
        self.driver.reset("episode-1")
        self.driver.apply((1.5, -0.5))
        np.testing.assert_array_equal(self.world.targets, [[-0.5, 1.5]])
        self.assertEqual(self.driver.observe().state, (0.75, 0.25))
        self.driver.step()
        observed = self.driver.observe()
        self.assertEqual(observed.state, (1.5, -0.5))
        self.assertEqual(observed.step, 1)
        self.assertAlmostEqual(observed.sim_time, 1.0 / _SPEC.fps)
        self.world.manager.step.assert_called_once_with(steps=2, update_fabric=True)

    def test_rejects_invalid_targets(self):
        self.driver.reset("episode-1")
        for targets in ((0.0,), (0.0, float("nan")), (float("inf"), 0.0), (2.1, 0.0), (0.0, -1.1)):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.driver.apply(targets)
        self.world.robot.set_dof_position_targets.assert_not_called()

    def test_reset_restores_stage(self):
        self.driver.reset("episode-1")
        self.driver.apply((1.5, -0.5))
        self.driver.step()
        self.assertEqual(self.world.cup_velocity, 1.0)
        self.driver.reset("episode-2")
        observed = self.driver.observe()
        self.assertEqual(
            (observed.episode_id, observed.step, observed.sim_time), ("episode-2", 0, 0.0)
        )
        self.assertEqual(observed.state, (0.75, 0.25))
        self.assertEqual((self.world.cup_position, self.world.cup_velocity), (0.0, 0.0))
        np.testing.assert_array_equal(self.world.targets, self.world.positions)
        self.assertEqual(self.world.context.open_stage.call_count, 2)
        self.assertEqual(self.world.stage.Reload.call_count, 2)
        self.world.manager.invalidate_physics.assert_called_once()

    def test_rejects_backend_joints(self):
        for joints in (("elbow", "other"), ("elbow", "elbow")):
            with self.subTest(joints=joints), self.assertRaisesRegex(ValueError, "joints"):
                self.world.robot.shared_metatype.dof_names = joints
                self.driver.reset("episode-1")

    def test_rejects_clock_drift(self):
        self.driver.reset("episode-1")
        self.world.rep.orchestrator.step.side_effect = lambda **kwargs: setattr(
            self.world, "time", 1.0
        )
        with self.assertRaisesRegex(RuntimeError, "timestep"):
            self.driver.observe()

    def test_rejects_wrong_step_dt(self):
        self.driver.reset("episode-1")
        self.world.manager.step.side_effect = lambda **kwargs: setattr(self.world, "time", 1.0)
        with self.assertRaisesRegex(RuntimeError, "timestep"):
            self.driver.step()

    def test_requires_episode(self):
        with self.assertRaisesRegex(RuntimeError, "Reset"):
            self.driver.observe()
        with self.assertRaisesRegex(RuntimeError, "Reset"):
            self.driver.apply((0.0, 0.0))
        with self.assertRaisesRegex(RuntimeError, "Reset"):
            self.driver.step()

    def test_rejects_invalid_pixels(self):
        self.driver.reset("episode-1")
        self.world.rep.annotators.get.return_value.get_data.return_value = np.zeros((2, 2, 3))
        with self.assertRaisesRegex(RuntimeError, "RGB"):
            self.driver.observe()

    def test_main_thread_only(self):
        errors = []

        def observe():
            try:
                self.driver.observe()
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=observe)
        thread.start()
        thread.join()
        self.assertEqual(errors, ["Isaac must run on the main thread"])


class IsaacSpecTests(unittest.TestCase):
    def test_rejects_invalid_spec(self):
        for change in (
            {"joints": ("pan", "pan")},
            {"joints": ()},
            {"physics_hz": 59},
            {"fps": 0},
            {"width": -1},
            {"height": True},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                IsaacSim(replace(_SPEC, **change))

    def test_requires_license(self):
        with (
            patch.dict(os.environ, {"ACCEPT_EULA": ""}),
            self.assertRaisesRegex(RuntimeError, "EULA"),
        ):
            IsaacSim(_SPEC).__enter__()


class IsaacStartupTests(unittest.TestCase):
    def test_sdk_error_before_close(self):
        events = []
        app = MagicMock()
        app.close.side_effect = lambda **kwargs: events.append("close")
        sdk = SimpleNamespace(SimulationApp=MagicMock(return_value=app))

        def fail_loading(driver):
            events.append("load")
            raise ModuleNotFoundError("Missing simulation extension")

        with (
            patch.dict(sys.modules, {"isaacsim": sdk}),
            patch.dict(os.environ, {"ACCEPT_EULA": "Y"}),
            patch.object(IsaacSim, "_load_sdk", fail_loading),
        ):
            with IsaacSim(_SPEC) as simulation:
                events.append("entered")
                with self.assertRaisesRegex(ModuleNotFoundError, "extension"):
                    simulation.reset("episode")
                events.append("failure-recorded")

        self.assertEqual(events, ["entered", "load", "failure-recorded", "close"])


if __name__ == "__main__":
    unittest.main()
