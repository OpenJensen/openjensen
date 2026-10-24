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
REGISTERS = {
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
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
        return max(0.0, min(100.0, (raw - low) * 100 / (high - low)))
    return (raw - (low + high) / 2) * 360 / 4095


def denormalized(joint: str, value: float, calibration: dict) -> int:
    if not math.isfinite(value):
        raise ValueError("Joint target must be finite")
    entry = calibration[joint]
    low, high = entry["range_min"], entry["range_max"]
    raw = (
        value * (high - low) / 100 + low
        if joint == "gripper"
        else value * 4095 / 360 + (low + high) / 2
    )
    return max(low, min(high, int(raw)))


class ServoBus:
    """Each instance owns one port; callers serialize all access through .lock."""

    def __init__(self, port: str, baudrate: int = 1_000_000):
        import scservo_sdk as sdk

        self.sdk = sdk
        self.port = sdk.PortHandler(port)
        self.packet = sdk.PacketHandler(0)
        self.lock = threading.RLock()
        self.opened = False
        self.corrupt = False
        try:
            if not self.port.openPort() or not self.port.setBaudRate(baudrate):
                raise HardwareError("Could not open serial port at requested baudrate")
            self.opened = True
            # Bound all packet waits: fast enough for the hardware watchdog to stop.
            original_timeout = self.port.setPacketTimeout

            def bounded_timeout(length):
                original_timeout(length)
                self.port.setPacketTimeoutMillis(25)

            self.port.setPacketTimeout = bounded_timeout
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
        # persistent communication failures or any motor fault still stop.
        for attempt in range(3):
            request = [0, 0, motor_id, 4, 2, address, size, 0]
            try:
                reply, result, error = self.packet.txRxPacket(self.port, request)
            except IndexError:
                # A malformed frame can also fail inside the SDK parser.
                self.port.is_using = False
                reply, result, error = [], self.sdk.COMM_RX_CORRUPT, 0
            value = 0
            if result == self.sdk.COMM_SUCCESS and not error:
                if (
                    not isinstance(reply, list)
                    or len(reply) != size + 6
                    or reply[:2] != [255, 255]
                    or reply[2] != motor_id
                    or reply[3] != size + 2
                    or reply[-1] != (~sum(reply[2:-1]) & 255)
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
        self._check(result, error, motor_id, f"read {register}")
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
        method = self.packet.write1ByteTxRx if size == 1 else self.packet.write2ByteTxRx
        result, error = method(self.port, motor_id, address, int(value))
        if (
            result == self.sdk.COMM_RX_TIMEOUT
            and not error
            and register
            in {
                "Homing_Offset",
                "Min_Position_Limit",
                "Max_Position_Limit",
                "Operating_Mode",
                "Lock",
            }
        ):
            # A lost acknowledgement does not mean the write failed. Verify
            # configuration without replaying a write or any motion command.
            time.sleep(0.01)
            try:
                actual = self.read(register, motor_id)
            except HardwareError as caught:
                raise HardwareError(
                    f"Motor {motor_id} write {register} acknowledgement missing; "
                    f"read-back failed: {caught}"
                ) from caught
            if actual == requested:
                return
            raise HardwareError(
                f"Motor {motor_id} write {register} acknowledgement missing; "
                f"read-back was {actual}, expected {requested}"
            )
        self._check(result, error, motor_id, f"write {register}")

    def _check(self, result, error, motor_id, operation):
        if result != self.sdk.COMM_SUCCESS or error:
            reason = (
                self.packet.getTxRxResult(result) if result else self.packet.getRxPacketError(error)
            )
            raise HardwareError(f"Motor {motor_id} {operation} failed: {reason}")

    def positions(self):
        with self.lock:
            return {
                joint: self.read("Present_Position", motor_id) for joint, motor_id in IDS.items()
            }

    def disable(self):
        # Attempt every motor even if a previous one has disconnected.
        errors = []
        for motor_id in IDS.values():
            try:
                self.write("Torque_Enable", motor_id, 0)
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
                "id": motor_id,
                "drive_mode": 0,
                "homing_offset": self.read("Homing_Offset", motor_id),
                "range_min": self.read("Min_Position_Limit", motor_id),
                "range_max": self.read("Max_Position_Limit", motor_id),
            }
            for joint, motor_id in IDS.items()
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


def probe_bus(bus, ids=range(1, 7), repeats=3):
    motors = []
    for motor_id in ids:
        replies = [bus.ping(motor_id) for _ in range(repeats)]
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
        "warning": ISOLATION_WARNING,
    }
