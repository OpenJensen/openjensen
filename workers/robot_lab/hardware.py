"""Small Feetech STS3215 adapter; no ML framework imports or implicit torque changes.

Register addresses and normalization match LeRobot v0.4.4's Feetech tables.
https://github.com/huggingface/lerobot/tree/v0.4.4/src/lerobot/motors
"""

from __future__ import annotations

import math
import threading
import time

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
IDS = {name: index + 1 for index, name in enumerate(JOINTS)}
BAUDRATES = (1_000_000, 500_000, 250_000, 128_000, 115_200, 57_600, 38_400, 19_200)
DISCOVERY_TIMEOUT_SECONDS = 20.0
REGISTERS = {
    "ID": (5, 1),
    "Return_Delay_Time": (7, 1),
    "Maximum_Acceleration": (85, 1),
    "Baud_Rate": (6, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Phase": (18, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Max_Torque_Limit": (16, 2),
    "Protection_Current": (28, 2),
    "Overload_Torque": (36, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Velocity": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
}
ISOLATION_WARNING = (
    "An ID scan cannot distinguish multiple factory motors sharing ID 1. Physically "
    "connect only the selected motor before assigning its ID."
)


class HardwareError(RuntimeError):
    pass


def decode_sign(value: int, bit: int = 15) -> int:
    return -(value & ((1 << bit) - 1)) if value & (1 << bit) else value


def encode_sign(value: int, bit: int = 11) -> int:
    return abs(value) | ((1 << bit) if value < 0 else 0)


def normalized(joint: str, raw: int, calibration: dict) -> float:
    entry = calibration[joint]
    low, high = entry["range_min"], entry["range_max"]
    if joint == "gripper":
        return (min(high, max(low, raw)) - low) / (high - low) * 100
    return (raw - (low + high) / 2) * 360 / 4095


def denormalized(joint: str, value: float, calibration: dict) -> int:
    if not math.isfinite(value):
        raise ValueError("Joint target must be finite")
    entry = calibration[joint]
    low, high = entry["range_min"], entry["range_max"]
    raw = (
        (min(100.0, max(0.0, value)) / 100) * (high - low) + low
        if joint == "gripper"
        else value * 4095 / 360 + (low + high) / 2
    )
    return max(low, min(high, int(raw)))


class ServoBus:
    """Each instance owns one port; callers serialize all access through .lock."""

    joints = JOINTS

    def __init__(self, port: str, baudrate: int = 1_000_000):
        import scservo_sdk as sdk

        self.sdk = sdk
        self.port = sdk.PortHandler(port)
        self.packet = sdk.PacketHandler(0)
        self.lock = threading.RLock()
        self.opened = False
        self.corrupt = False
        self.gripper_load_tolerance = False
        self.motor_status = {}
        try:
            if not self.port.openPort() or not self.port.setBaudRate(baudrate):
                raise HardwareError("Could not open serial port at requested baudrate")
            self.opened = True
            # Use the same Feetech SDK timing fix as the upstream SO follower.
            from lerobot_control import patch_setPacketTimeout

            self.port.setPacketTimeout = patch_setPacketTimeout.__get__(self.port, type(self.port))
        except Exception:
            try:
                self.port.closePort()
            except Exception:
                pass
            raise

    def close(self):
        with self.lock:
            if self.opened:
                self.port.closePort()
                self.opened = False

    def ping(self, motor_id: int):
        model, result, error = self.packet.ping(self.port, motor_id)
        if result == self.sdk.COMM_RX_CORRUPT:
            self.corrupt = True
            raise HardwareError(
                "Corrupt motor replies: disconnect the other motors and check cabling/power"
            )
        if result != self.sdk.COMM_SUCCESS:
            return None
        if error:
            raise HardwareError(
                f"Motor {motor_id} reports an error: {self.packet.getRxPacketError(error)}"
            )
        return int(model)

    def read(self, register: str, motor_id: int):
        address, size = REGISTERS[register]
        # Validate the status packet before decoding. The SDK convenience reads
        # slice a short write ACK as if it contained data (including its checksum),
        # returning a bogus byte or raising IndexError for a two-byte register.
        # Reads are safe to repeat. The SDK reports both truncated replies and
        # checksum failures as RX_CORRUPT. Discard them and request fresh data;
        # persistent communication failures and unhandled motor faults still stop.
        for attempt in range(3):
            request = [0, 0, motor_id, 4, 2, address, size, 0]
            try:
                reply, result, error = self.packet.txRxPacket(self.port, request)
            except IndexError:
                # A malformed frame can also fail inside the SDK parser.
                self.port.is_using = False
                reply, result, error = [], self.sdk.COMM_RX_CORRUPT, 0
            value = 0
            if result == self.sdk.COMM_SUCCESS:
                if (
                    not isinstance(reply, list)
                    or len(reply) != size + 6
                    or reply[:2] != [255, 255]
                    or reply[2] != motor_id
                    or reply[3] != size + 2
                    or reply[-1] != (~sum(reply[2:-1]) & 255)
                    or reply[4] != error
                ):
                    result = self.sdk.COMM_RX_CORRUPT
                else:
                    error = reply[4]
                    value = int.from_bytes(bytes(reply[5:-1]), "little")
            if (
                result not in (self.sdk.COMM_RX_TIMEOUT, self.sdk.COMM_RX_CORRUPT)
                or error
                or attempt == 2
            ):
                break
            time.sleep(0.005)
            # SDK clearPort() only flushes output. Remove a partial/stale input
            # packet before sending the next read, never reuse its value.
            self.port.ser.reset_input_buffer()
        # An overload alarm is feedback, not a rejected read, in current
        # LeRobot. Admit it only with a complete, validated gripper packet.
        if result == self.sdk.COMM_SUCCESS and register == "Present_Position":
            if not hasattr(self, "motor_status"):
                self.motor_status = {}
            self.motor_status[motor_id] = error
        self._check(
            result,
            error,
            motor_id,
            f"read {register}",
            allow_gripper_load=(
                register == "Torque_Enable"
                or (
                    register == "Present_Position"
                    and getattr(self, "gripper_load_tolerance", False)
                )
            ),
        )
        if register == "Homing_Offset":
            return decode_sign(int(value), 11)
        if register == "Present_Position":
            return decode_sign(int(value), 15)
        return int(value)

    def write(self, register: str, motor_id: int, value: int):
        address, size = REGISTERS[register]
        requested = int(value)
        if register == "Homing_Offset":
            value = encode_sign(value)
        # Check the complete write ACK too: a delayed read response must not
        # stand in for acknowledgement of a later command to the same motor.
        request = [0, 0, motor_id, size + 3, 3, address, *int(value).to_bytes(size, "little"), 0]
        try:
            reply, result, error = self.packet.txRxPacket(self.port, request)
        except IndexError:
            self.port.is_using = False
            reply, result, error = [], self.sdk.COMM_RX_CORRUPT, 0
        if result == self.sdk.COMM_SUCCESS:
            if (
                not isinstance(reply, list)
                or len(reply) != 6
                or reply[:2] != [255, 255]
                or reply[2] != motor_id
                or reply[3] != 2
                or reply[-1] != (~sum(reply[2:-1]) & 255)
                or reply[4] != error
            ):
                result = self.sdk.COMM_RX_CORRUPT
            else:
                error = reply[4]
        if (
            result in (self.sdk.COMM_RX_TIMEOUT, self.sdk.COMM_RX_CORRUPT)
            and not error
            and register
            in {
                "Homing_Offset",
                "Min_Position_Limit",
                "Max_Position_Limit",
                "Operating_Mode",
                "Phase",
                "Lock",
                "Max_Torque_Limit",
                "Torque_Limit",
                "Protection_Current",
                "Overload_Torque",
                "Acceleration",
                "Maximum_Acceleration",
                "Return_Delay_Time",
                "P_Coefficient",
                "I_Coefficient",
                "D_Coefficient",
                "Torque_Enable",
            }
        ):
            # A damaged/missing ACK does not establish whether the command
            # took effect. Verify the register with a new validated read, never
            # replay the write. Torque read-back resolves the actual on/off
            # state; target, velocity, ID and baud writes remain fail-closed.
            time.sleep(0.01)
            self.port.ser.reset_input_buffer()
            try:
                actual = self.read(register, motor_id)
            except HardwareError as caught:
                raise HardwareError(
                    f"Motor {motor_id} write {register} acknowledgement unverified; "
                    f"read-back failed: {caught}"
                ) from caught
            if actual == requested:
                return
            raise HardwareError(
                f"Motor {motor_id} write {register} acknowledgement unverified; "
                f"read-back was {actual}, expected {requested}"
            )
        self._check(
            result,
            error,
            motor_id,
            f"write {register}={requested}",
            # An alarm does not establish that torque-off was refused. A fresh,
            # validated zero read-back below still has to prove it took effect.
            allow_gripper_load=register == "Torque_Enable" and requested == 0,
        )
        if register == "Torque_Enable" and self.read(register, motor_id) != requested:
            raise HardwareError(
                f"Motor {motor_id} torque read-back did not match requested state {requested}"
            )

    def write_positions(self, values, is_current=lambda: True):
        """Use LeRobot's GroupSyncWrite position path: one packet, no ACK per joint."""
        with self.lock:
            if not self.opened or not is_current():
                raise HardwareError("Control session stopped before writing goals")
            if not values or set(values) - set(self.joints):
                raise ValueError("Goal packet must contain only selected joints")
            writer = self.sdk.GroupSyncWrite(self.port, self.packet, *REGISTERS["Goal_Position"])
            try:
                for joint, value in values.items():
                    if type(value) is not int or not 0 <= value <= 4095:
                        raise ValueError("Goal position must be an encoder count from 0 to 4095")
                    if not writer.addParam(IDS[joint], list(value.to_bytes(2, "little"))):
                        raise HardwareError(f"Could not prepare {joint} position packet")
                if not self.opened or not is_current():
                    raise HardwareError("Control session stopped before sending goals")
                self._check(writer.txPacket(), 0, "selected", "sync write Goal_Position")
            finally:
                writer.clearParam()

    def _check(self, result, error, motor_id, operation, *, allow_gripper_load=False):
        # Only gripper overload is tolerated. Other joints, mixed/unknown flags,
        # temperature, current, voltage and encoder faults still stop control.
        if (
            result == self.sdk.COMM_SUCCESS
            and error == 0x20
            and motor_id == IDS["gripper"]
            and allow_gripper_load
        ):
            return
        if result != self.sdk.COMM_SUCCESS or error:
            reason = (
                self.packet.getTxRxResult(result) if result else self.packet.getRxPacketError(error)
            )
            fault = HardwareError(f"Motor {motor_id} {operation} failed: {reason}")
            fault.details = {
                "code": "motor_status_fault" if error else "motor_communication_fault",
                "motor_id": motor_id,
                "operation": operation,
                "communication_result": result,
                "status_error": error,
            }
            raise fault

    def positions(self):
        with self.lock:
            return {joint: self.read("Present_Position", IDS[joint]) for joint in self.joints}

    def disable(self):
        # Attempt every motor even if a previous one has disconnected.
        errors = []
        for joint in self.joints:
            try:
                self.write("Torque_Enable", IDS[joint], 0)
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise HardwareError("; ".join(errors))

    def check_six(self):
        results = probe_bus(self)
        if not results["ready"]:
            raise HardwareError(
                "All six STS3215 motors must respond reliably at IDs 1–6. Complete Motor IDs first."
            )

    def read_calibration(self):
        return {
            joint: {
                "id": IDS[joint],
                "drive_mode": 0,
                "homing_offset": self.read("Homing_Offset", IDS[joint]),
                "range_min": self.read("Min_Position_Limit", IDS[joint]),
                "range_max": self.read("Max_Position_Limit", IDS[joint]),
            }
            for joint in self.joints
        }

    def write_calibration(self, calibration):
        for joint, entry in calibration.items():
            motor_id = IDS[joint]
            write_error = None
            try:
                self.write("Lock", motor_id, 0)
                for register, key in (
                    ("Homing_Offset", "homing_offset"),
                    ("Min_Position_Limit", "range_min"),
                    ("Max_Position_Limit", "range_max"),
                ):
                    self.write(register, motor_id, entry[key])
            except Exception as error:
                write_error = error
                raise
            finally:
                # Lock=1 persists the EEPROM changes. Even a failed/uncertain
                # unlock or field write must make this bounded re-lock attempt.
                try:
                    self.write("Lock", motor_id, 1)
                except Exception as error:
                    if write_error is not None:
                        raise HardwareError(
                            f"Calibration write failed ({write_error}); EEPROM re-lock "
                            f"also failed ({error}) for motor {motor_id}"
                        ) from write_error
                    raise


def probe_bus(bus, ids=range(1, 7), repeats=3, *, discover=False, is_current=lambda: True):
    motors = []
    deadline = time.monotonic() + DISCOVERY_TIMEOUT_SECONDS if discover else None
    if discover:
        ids = range(254)

    def ping(motor_id):
        if not is_current():
            raise HardwareError("Motor scan cancelled by Stop")
        if deadline is not None and time.monotonic() >= deadline:
            raise HardwareError("Motor discovery exceeded its 20-second budget; scan is incomplete")
        return bus.ping(motor_id)

    for motor_id in ids:
        replies = [ping(motor_id)]
        # Discovery is bounded to one timeout per absent address. Confirm each
        # responding address with the same repeated checks as a normal scan.
        if not discover or replies[0] is not None:
            replies.extend(ping(motor_id) for _ in range(repeats - 1))
        found = [reply for reply in replies if reply is not None]
        if found:
            motors.append(
                {
                    "id": motor_id,
                    "model": found[0],
                    "stable": len(found) == repeats and len(set(found)) == 1,
                    "responses": len(found),
                }
            )
    ready = (
        len(motors) == 6
        and {entry["id"] for entry in motors} == set(IDS.values())
        and all(entry["stable"] and entry["model"] == 777 for entry in motors)
    )
    return {
        "motors": motors,
        "ready": ready,
        "expected_ids": list(IDS.values()),
        "scan_scope": "all_addresses" if discover else "expected_ids",
        "unexpected_ids": [entry["id"] for entry in motors if entry["id"] not in IDS.values()],
        "warning": ISOLATION_WARNING,
    }


def check_selected_motors(bus, joints):
    expected = {IDS[j] for j in joints}
    result = probe_bus(bus, sorted(expected))
    reliable = {
        motor["id"] for motor in result["motors"] if motor["stable"] and motor["model"] == 777
    }
    if reliable != expected:
        missing = ", ".join(str(i) for i in sorted(expected - reliable))
        raise HardwareError(f"Selected motor IDs {missing} did not respond reliably")
