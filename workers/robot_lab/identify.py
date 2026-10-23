"""Explicit, bounded identification motion, separate from motion-free ID splitting.

STS3215 SRAM registers match LeRobot v0.4.4's Feetech control table. No EEPROM,
ID, calibration or broadcast torque-on writes are issued here. The caller owns
its existing transport and I/O lock, and has verified the six-address inventory.
"""

from __future__ import annotations

import time

from _vendor.servotools.framing import find_frame, packet


class IdentificationUnavailable(ValueError):
    pass


AMPLITUDE = 24  # 2.1 degrees; fixed by the server, never supplied by the browser.


def wiggle(bus, motor_id, ids, stop_event, *, motion_state=lambda active: None):
    if type(motor_id) is not int or motor_id not in ids or len(set(ids)) != 6:
        raise ValueError("Verify six distinct motors and select one address first")
    deadline = time.monotonic() + 3.0

    def check():
        if stop_event.is_set():
            raise ValueError("Identification motion stopped")
        if time.monotonic() >= deadline:
            raise ValueError("Identification motion exceeded its time budget")

    def read(address, size, *, cleanup=False):
        if not cleanup:
            check()
        request = packet(motor_id, 2, bytes([address, size]))
        raw = bus.exchange(request, 0.001, 64)
        if raw.startswith(request):
            raw = raw[len(request) :]
        data = find_frame(raw, motor_id, size, check_error=True)
        if data is None:
            raise ValueError(f"Motor {motor_id} identification read-back failed")
        return int.from_bytes(data, "little")

    def write(address, value, size=1):
        # Finite SRAM-only write surface. Restoration uses the same registers;
        # the selected address cannot be changed or replaced by broadcast.
        if address not in {40, 41, 42, 44, 46, 48}:
            raise ValueError("Register not permitted for identification motion")
        bus.send(packet(motor_id, 3, bytes([address]) + value.to_bytes(size, "little")))
        bus.flush()
        time.sleep(0.005)

    check()
    if read(33, 1) != 0 or read(40, 1) != 0:
        raise IdentificationUnavailable(
            "Identification requires position mode with torque released"
        )
    center = read(56, 2)
    low, high = read(9, 2), read(11, 2)
    if not 0 <= low < high <= 4095 or not low + AMPLITUDE <= center <= high - AMPLITUDE:
        raise IdentificationUnavailable(
            "Reposition every joint clear of its stops before identification"
        )
    original = {
        address: read(address, size) for address, size in ((41, 1), (44, 2), (46, 2), (48, 2))
    }
    if not 1 <= original[48] <= 1000:
        raise IdentificationUnavailable(
            "Motor torque limit is unavailable; use hand identification"
        )
    observed = [center]
    cleanup_errors = []
    try:
        check()
        write(48, min(original[48], 100), 2)
        write(41, 5)
        write(44, 0, 2)
        write(46, 100, 2)
        write(42, center, 2)
        if read(48, 2) > 100 or read(42, 2) != center:
            raise ValueError("Identification limits could not be verified")
        check()
        # Report conservatively before the possibly accepted torque-on packet.
        motion_state(True)
        write(40, 1)
        for target in (center + AMPLITUDE, center - AMPLITUDE, center):
            check()
            write(42, target, 2)
            for _ in range(5):
                if stop_event.wait(0.05):
                    check()
                current = read(56, 2)
                if not center - AMPLITUDE - 40 <= current <= center + AMPLITUDE + 40:
                    raise ValueError("Unexpected joint displacement; identification stopped")
                observed.append(current)
    finally:
        # Cleanup remains mandatory after cancellation or an uncertain USB write.
        torque_off = False
        try:
            write(40, 0)
            torque_off = read(40, 1, cleanup=True) == 0
            if not torque_off:
                raise ValueError("Torque-off was not verified")
        except Exception as error:
            cleanup_errors.append(str(error))
        if torque_off:
            motion_state(False)
            for address, size in ((41, 1), (44, 2), (46, 2), (48, 2)):
                try:
                    write(address, original[address], size)
                    if read(address, size, cleanup=True) != original[address]:
                        raise ValueError(f"SRAM register {address} restoration failed")
                except Exception as error:
                    cleanup_errors.append(str(error))
        if cleanup_errors:
            raise ValueError("Identification cleanup failed: " + "; ".join(cleanup_errors))
    movement = max(observed) - min(observed)
    if movement < 4:
        raise IdentificationUnavailable(
            "No measurable wiggle at low torque; use hand identification"
        )
    return movement
