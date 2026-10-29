# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# See LEROBOT-LICENSE and LEROBOT-PROVENANCE.json in this directory.
"""Pinned LeRobot SO follower methods with the companion's gated hardware bridge.

configure(), send_action(), configure_motors(), the target-clamping helper and
packet-timeout patch are copied from LeRobot v0.4.4. Their ASTs are recorded in
the provenance file. The bridge keeps configuration torque-off until OpenJensen
has verified calibration and current-pose hold targets on every selected motor.
"""

# ruff: noqa: E501

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from enum import Enum
from pprint import pformat
from types import SimpleNamespace

from hardware import IDS, HardwareError, denormalized, normalized

RobotAction = dict[str, float]


class OperatingMode(Enum):
    POSITION = 0


def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]], max_relative_target: float | dict[str, float]
) -> dict[str, float]:
    """Caps relative action target magnitude for safety."""

    if isinstance(max_relative_target, float):
        diff_cap = dict.fromkeys(goal_present_pos, max_relative_target)
    elif isinstance(max_relative_target, dict):
        if not set(goal_present_pos) == set(max_relative_target):
            raise ValueError("max_relative_target keys must match those of goal_present_pos.")
        diff_cap = max_relative_target
    else:
        raise TypeError(max_relative_target)

    warnings_dict = {}
    safe_goal_positions = {}
    for key, (goal_pos, present_pos) in goal_present_pos.items():
        diff = goal_pos - present_pos
        max_diff = diff_cap[key]
        safe_diff = min(diff, max_diff)
        safe_diff = max(safe_diff, -max_diff)
        safe_goal_pos = present_pos + safe_diff
        safe_goal_positions[key] = safe_goal_pos
        if abs(safe_goal_pos - goal_pos) > 1e-4:
            warnings_dict[key] = {
                "original goal_pos": goal_pos,
                "safe goal_pos": safe_goal_pos,
            }

    if warnings_dict:
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"{pformat(warnings_dict, indent=4)}"
        )

    return safe_goal_positions


def patch_setPacketTimeout(self, packet_length):  # noqa: N802
    """
    HACK: This patches the PortHandler behavior to set the correct packet timeouts.

    It fixes https://gitee.com/ftservo/SCServoSDK/issues/IBY2S6
    The bug is fixed on the official Feetech SDK repo (https://gitee.com/ftservo/FTServo_Python)
    but because that version is not published on PyPI, we rely on the (unofficial) on that is, which needs
    patching.
    """
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (
        (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50
    )


class UpstreamSOFollower:
    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.bus.write(
                        "Max_Torque_Limit", motor, 500
                    )  # 50% of max torque to avoid burnout
                    self.bus.write(
                        "Protection_Current", motor, 250
                    )  # 50% of max current to avoid burnout
                    self.bus.write("Overload_Torque", motor, 25)

    def send_action(self, action: RobotAction) -> RobotAction:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            RobotAction: the action sent to the motors, potentially clipped.
        """

        goal_pos = {
            key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")
        }

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}


class MotorBridge:
    def configure_motors(
        self, return_delay_time=0, maximum_acceleration=254, acceleration=254
    ) -> None:
        for motor in self.motors:
            # By default, Feetech motors have a 500µs delay response time (corresponding to a value of 250 on
            # the 'Return_Delay_Time' address). We ensure this is reduced to the minimum of 2µs (value of 0).
            self.write("Return_Delay_Time", motor, return_delay_time)
            # Set 'Maximum_Acceleration' to 254 to speedup acceleration and deceleration of the motors.
            if self.protocol_version == 0:
                self.write("Maximum_Acceleration", motor, maximum_acceleration)
            self.write("Acceleration", motor, acceleration)

    def __init__(self, bus, calibration, is_current):
        self.bus = bus
        self.calibration = calibration
        self.motors = dict.fromkeys(calibration)
        self.protocol_version = 0
        self.is_current = is_current
        self.last_raw = {}

    def require_current(self):
        if not self.is_current() or not self.bus.opened:
            raise HardwareError("Control session stopped; no further goals sent")

    @contextmanager
    def torque_disabled(self):
        self.require_current()
        self.bus.disable()
        for joint in self.motors:
            self.require_current()
            self.bus.write("Lock", IDS[joint], 0)
        # Unlike the upstream standalone context, do not automatically energize
        # on exit (including on exceptions). control_start owns that final gate.
        yield

    def write(self, register, joint, value):
        self.require_current()
        motor_id = IDS[joint]
        self.bus.write(register, motor_id, value)
        self.require_current()
        if self.bus.read(register, motor_id) != value:
            raise HardwareError(
                f"Motor {motor_id} {register} read-back did not match; motion remains disabled"
            )

    def sync_read(self, register):
        self.require_current()
        if register != "Present_Position":
            raise ValueError("Only joint positions can be read through the control bridge")
        return {
            joint: normalized(joint, value, self.calibration)
            for joint, value in self.bus.positions().items()
        }

    def sync_write(self, register, values):
        self.require_current()
        if register != "Goal_Position" or not values or set(values) - self.motors.keys():
            raise ValueError("Control goals must name selected joints")
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values.values()):
            raise ValueError("Control goals must be finite positions")
        raw = {
            joint: denormalized(joint, value, self.calibration) for joint, value in values.items()
        }
        raw = {
            joint: max(
                self.calibration[joint]["range_min"] + 8,
                min(self.calibration[joint]["range_max"] - 8, value),
            )
            for joint, value in raw.items()
        }
        self.require_current()
        self.bus.write_positions(raw, self.is_current)
        self.require_current()
        self.last_raw.update(raw)


class LeRobotPositionControl:
    """Run upstream SO follower setup/actions without the ML training dependencies."""

    def __init__(self, bus, calibration, is_current):
        self.bridge = MotorBridge(bus, calibration, is_current)
        self.follower = UpstreamSOFollower()
        self.follower.bus = self.bridge
        self.follower.config = SimpleNamespace(max_relative_target=None)
        self.goals = {}
        self.torque_limits = {}

    def configure(self):
        self.bridge.bus.gripper_load_tolerance = False
        self.follower.configure()
        # Earlier companions leave a 100-tick/s speed cap in the volatile goal
        # register. Restore the position-mode zero/default before enabling.
        for joint in self.bridge.motors:
            self.bridge.write("Goal_Velocity", joint, 0)
            self.bridge.require_current()
            # The ST servo copies EEPROM maximum torque (16) into RAM limit
            # (48) at power-on. Clear an interrupted identification's 10% RAM
            # cap using that same saved ceiling, never a higher invented limit.
            limit = (
                500 if joint == "gripper" else self.bridge.bus.read("Max_Torque_Limit", IDS[joint])
            )
            if type(limit) is not int or not 1 <= limit <= 1000:
                raise HardwareError(
                    f"{joint} saved maximum torque is invalid; motion remains disabled"
                )
            self.bridge.write("Torque_Limit", joint, limit)
            self.torque_limits[joint] = limit
        # Arm load-alarm handling after all upstream gripper protection
        # settings and the runtime torque ceiling have passed read-back.
        self.bridge.bus.gripper_load_tolerance = "gripper" in self.bridge.motors

    def command(self, positions):
        action = {joint + ".pos": value for joint, value in positions.items()}
        self.follower.send_action(action)
        self.goals.update(action)
        return {joint: self.bridge.last_raw[joint] for joint in positions}

    def command_raw(self, positions):
        return self.command(
            {
                joint: normalized(joint, value, self.bridge.calibration)
                for joint, value in positions.items()
            }
        )

    def repeat(self):
        if self.goals:
            self.follower.send_action(self.goals)

    def hold(self, raw):
        # control_start already wrote and verified these exact raw hold targets.
        self.bridge.last_raw = dict(raw)
        self.goals = {
            joint + ".pos": normalized(joint, value, self.bridge.calibration)
            for joint, value in raw.items()
        }


PROFILE = {
    "name": "LeRobot SO101 position control",
    "upstream": "huggingface/lerobot@v0.4.4",
    "target_mode": "direct_position",
    "max_relative_target": None,
    "acceleration": 254,
    "maximum_acceleration": 254,
    "goal_velocity": 0,
    "gripper_load_response": "continue_with_motor_protection",
    "observation_range_policy": "lerobot_normalization",
    "body_position_units": "degrees",
    "gripper_position_units": "percent",
    "recording_camera": "latest_frame",
    "recording_images": "background_writer",
}
