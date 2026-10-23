"""Experimental assembled STS3215 identity setup, without position/torque-on commands.

Reboot-wake splitting is adapted from Enigma-Incorporated/servotools, commit
1c65c57587f91085cd32fca3743f3e6f464cfe35 (MIT; see _vendor/servotools/LICENSE).
This is not a manufacturer-supported unique-serial addressing mechanism.
Nothing runs on import/construction. The server supplies physical support and
write acknowledgements and reserves the port before calling split/apply.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

from _vendor.servotools import registers as reg
from _vendor.servotools.framing import find_frame, packet
from dataset import atomic_json
from hardware import IDS, JOINTS

EXPECTED_COUNT = 6
MAX_ROUNDS = 12
PROBE_READS = 14
POSITION_MAX = 4096
MOVE_THRESHOLD = 80
DOMINANCE = 1.8


class AssembledError(RuntimeError):
    pass


class AssembledCancelled(AssembledError):
    pass


def circular_delta(a, b):
    delta = abs(int(a) - int(b)) % POSITION_MAX
    return min(delta, POSITION_MAX - delta)


def detect_movement(baseline, samples, assigned=(), now=None):
    """Return one dominant, fresh ID after 3/4 confirming manual-motion samples.

    samples are (monotonic_timestamp, {id: raw_position}) tuples. All samples
    must cover the exact same six IDs, be ordered/fresh, and contain valid raw
    encoders. Ambiguous motion or movement of an assigned joint returns None.
    """
    if len(baseline) != EXPECTED_COUNT or len(samples) < 4:
        return None
    now = time.monotonic() if now is None else now
    recent = samples[-4:]
    previous = None
    votes = []
    for captured_at, positions in recent:
        if (
            not isinstance(captured_at, (int, float))
            or not math.isfinite(captured_at)
            or captured_at > now + 0.05
            or now - captured_at > 1.5
            or previous is not None
            and captured_at <= previous
            or set(positions) != set(baseline)
        ):
            return None
        previous = captured_at
        if any(
            type(value) is not int or not 0 <= value < POSITION_MAX for value in positions.values()
        ):
            return None
        if any(
            type(value) is not int or not 0 <= value < POSITION_MAX for value in baseline.values()
        ):
            return None
        changes = {sid: circular_delta(positions[sid], baseline[sid]) for sid in baseline}
        ordered = sorted(changes, key=changes.get, reverse=True)
        first, runner = ordered[:2]
        if changes[first] < MOVE_THRESHOLD or first in assigned:
            votes.append(None)
        elif changes[first] < changes[runner] * DOMINANCE:
            votes.append(None)
        else:
            votes.append(first)
    candidates = {candidate for candidate in votes if candidate is not None}
    if len(candidates) != 1:
        return None
    candidate = candidates.pop()
    return candidate if votes.count(candidate) >= 3 else None


class SerialTransport:
    """Transport only; AssembledSetup validates every emitted packet."""

    def __init__(self, port):
        import serial

        self.serial = serial.Serial(
            port, 1_000_000, timeout=0.006, write_timeout=0.25, exclusive=True
        )

    @property
    def timeout(self):
        return self.serial.timeout

    @timeout.setter
    def timeout(self, value):
        self.serial.timeout = value

    def exchange(self, message, wait, length):
        self.serial.reset_input_buffer()
        self.serial.write(message)
        if wait:
            time.sleep(wait)
        return self.serial.read(length)

    def send(self, message):
        self.serial.reset_input_buffer()
        self.serial.write(message)

    def flush(self):
        # POSIX tcdrain (pyserial.flush) can wait forever on a disconnected
        # adapter. Poll its nonblocking queue counter with a fixed deadline.
        deadline = time.monotonic() + 0.25
        while self.serial.out_waiting:
            if time.monotonic() >= deadline:
                raise AssembledError("USB serial output did not drain within250ms")
            time.sleep(0.001)

    def close(self):
        self.serial.close()


class AssembledSetup:
    def __init__(
        self,
        port,
        data_dir,
        *,
        stop_event=None,
        io_lock=None,
        progress=None,
        bus_factory=None,
        timeout_s=240,
        verified_so101=False,
    ):
        self.port = port
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.io_lock = io_lock if io_lock is not None else threading.RLock()
        self.progress = progress
        self.bus_factory = bus_factory or SerialTransport
        self.timeout_s = min(600, max(10, float(timeout_s)))
        self.verified_so101 = verified_so101 is True
        self.unverified_models = []
        self.bus = None
        self.deadline = None
        self.reboot_cleanup_at = None
        self.lock = threading.RLock()
        directory = Path(data_dir) / "assembled"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.journal_path = directory / (hashlib.sha256(port.encode()).hexdigest()[:20] + ".json")
        if self.journal_path.is_symlink():
            raise AssembledError("The setup journal must not be a symbolic link")
        self.state = {
            "phase": "idle",
            "ids": [],
            "positions": {},
            "mapping": {},
            "round": 0,
            "message": "Ready for explicit assembled setup",
            "error": None,
            "pending": None,
            "journal_path": str(self.journal_path),
            "experimental": True,
            "port": port,
        }
        if self.journal_path.exists():
            if self.journal_path.stat().st_size > 256 * 1024:
                raise AssembledError("Setup journal is too large")
            saved = json.loads(self.journal_path.read_text())
            if saved.get("port") == port:
                self.state["previous"] = {
                    key: saved.get(key) for key in ("phase", "ids", "mapping", "pending", "message")
                }

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def _publish(self, journal=True, **updates):
        with self.lock:
            self.state.update(updates)
            snapshot = self.snapshot()
            if journal:
                atomic_json(self.journal_path, snapshot)
                self.journal_path.chmod(0o600)
                # Persist recovery intent before any next ID write, including
                # the rename itself. An fsync failure must fail the transaction
                # closed rather than claim a durable recovery record.
                with self.journal_path.open("rb") as stream:
                    os.fsync(stream.fileno())
                directory_fd = os.open(self.journal_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        if self.progress:
            self.progress(snapshot)
        return snapshot

    def _check(self):
        if self.stop_event.is_set():
            raise AssembledCancelled(
                "Assembled setup cancelled; partial IDs are preserved in its journal"
            )
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise AssembledError("Assembled setup exceeded its time budget; scan to resume safely")

    def _sleep(self, seconds):
        if self.stop_event.wait(max(0, seconds)):
            self._check()
        self._check()

    def _open(self):
        self._check()
        with self.io_lock:
            self._check()
            if self.bus is None:
                opened = self.bus_factory(self.port)
                try:
                    self._check()
                except Exception:
                    opened.close()
                    raise
                self.bus = opened

    @staticmethod
    def _validate(message):
        """Strict packet allowlist: reads, reboot, torque OFF, lock, ID only."""
        offset = 0
        while offset < len(message):
            if message[offset : offset + 2] != b"\xff\xff" or offset + 6 > len(message):
                raise AssembledError("Invalid assembled-setup packet")
            sid, length, instruction = message[offset + 2 : offset + 5]
            end = offset + 4 + length
            if end > len(message) or sid > 254 or length < 2:
                raise AssembledError("Invalid assembled-setup packet length")
            frame = message[offset:end]
            if (~sum(frame[2:-1]) & 255) != frame[-1]:
                raise AssembledError("Invalid assembled-setup packet checksum")
            params = frame[5:-1]
            if instruction == 0x02:
                if len(params) != 2 or params not in (
                    bytes([reg.PRESENT_POSITION, 2]),
                    bytes([reg.MODEL, 2]),
                    bytes([reg.ID, 1]),
                    bytes([reg.LOCK, 1]),
                    bytes([reg.TORQUE_ENABLE, 1]),
                ):
                    raise AssembledError("Only setup verification registers may be read")
            elif instruction == 0x08:
                if params or sid != 254:
                    raise AssembledError("Only whole-chain reboot is supported")
            elif instruction == 0x03:
                if len(params) != 2:
                    raise AssembledError("Setup writes must change one byte only")
                address, value = params
                if not (
                    (address == reg.TORQUE_ENABLE and value == 0)
                    or (address == reg.LOCK and value in (0, 1) and sid < 254)
                    or (address == reg.ID and 1 <= value <= 253 and sid < 254)
                ):
                    raise AssembledError(
                        "Assembled setup cannot command motion or change other motor settings"
                    )
            else:
                raise AssembledError("Instruction not allowed for assembled setup")
            offset = end
        if not message or len(message) > 84 * 16:
            raise AssembledError("Setup burst exceeds its bounded UART window")

    def _send(self, message, *, cleanup=False):
        self._validate(message)
        if cleanup and message not in (
            packet(message[2], 0x03, bytes([reg.LOCK, 1])),
            packet(message[2], 0x03, bytes([reg.TORQUE_ENABLE, 0])),
        ):
            raise AssembledError("Cancellation cleanup permits only EEPROM re-lock or torque OFF")
        with self.io_lock:
            if not cleanup:
                self._check()
            if self.bus is None:
                raise AssembledError("Setup serial connection is closed")
            self.bus.send(message)
            self.bus.flush()
            # out_waiting==0 can mean the USB bridge accepted the data while
            # its UART is still clocking it out. Hold the same Stop-critical
            # lock for the bounded nominal 8N1 wire duration plus USB guard.
            # This tiny drain interval is intentionally non-cancellable: Stop
            # waits here rather than reporting completion ahead of queued IDs.
            time.sleep(len(message) * 10 / 1_000_000 + 0.005)

    def _write(self, sid, address, value, *, cleanup=False):
        self._send(packet(sid, 0x03, bytes([address, value])), cleanup=cleanup)

    def _read_response(self, sid, address, length, *, wake_poll=False):
        request = packet(sid, 0x02, bytes([address, length]))
        self._validate(request)
        with self.io_lock:
            self._check()
            if self.bus is None:
                raise AssembledError("Setup serial connection is closed")
            response = self.bus.exchange(
                request, 0.0 if wake_poll else 0.001, 8 if wake_poll else 64
            )
        # Some adapters echo the transmit packet before the status frame.
        if response.startswith(request):
            response = response[len(request) :]
        # A valid servo error response is a hardware fault, not evidence of
        # duplicate IDs. Never reboot/rename a motor to hide a protection error.
        for index in range(max(0, len(response) - 5)):
            if response[index : index + 3] != bytes([255, 255, sid]):
                continue
            size = response[index + 3]
            end = index + 4 + size
            if size == length + 2 and end <= len(response):
                frame = response[index:end]
                if (~sum(frame[2:-1]) & 255) == frame[-1] and frame[4]:
                    raise AssembledError(
                        f"Motor {sid} reports status error {frame[4]}; resolve it before ID setup"
                    )
        return response, find_frame(response, sid, length, check_error=True)

    def _read(self, sid, address, length=1):
        _, data = self._read_response(sid, address, length)
        return int.from_bytes(data, "little") if data is not None else None

    def _classify(self, sid, repeats=PROBE_READS):
        clean = corrupt = 0
        raw_positions = []
        for _ in range(repeats):
            raw, value = self._read_response(sid, reg.PRESENT_POSITION, 2)
            if value is not None:
                position = int.from_bytes(value, "little")
                if not 0 <= position <= 4095:
                    corrupt += 1
                else:
                    raw_positions.append(position)
                    clean += 1
            elif raw:
                corrupt += 1
            self._sleep(0.001)
        return (
            "collision"
            if corrupt
            else "single"
            if clean == repeats
            else "unreliable"
            if clean
            else "empty"
        ), raw_positions

    def _scan(self):
        found = {}
        self.unverified_models = []
        for sid in range(0, 254):
            response, value = self._read_response(sid, reg.PRESENT_POSITION, 2)
            if not response and value is None:
                continue
            kind, _ = self._classify(sid)
            if kind != "empty":
                found[sid] = kind
                models = [self._read(sid, reg.MODEL, 2) for _ in range(3)]
                if 777 not in models:
                    self.unverified_models.append(sid)
                if any(model is not None and model != 777 for model in models):
                    raise AssembledError(
                        f"ID {sid} is not an STS3215 motor; no further setup writes allowed"
                    )
        self._publish(
            ids=sorted(found), collisions=[sid for sid, kind in found.items() if kind != "single"]
        )
        return found

    def _verify_id(self, sid):
        kind, _ = self._classify(sid)
        if kind != "single":
            raise AssembledError(f"ID {sid} does not have stable independent position replies")
        for _ in range(3):
            if self._read(sid, reg.MODEL, 2) != 777 or self._read(sid, reg.ID) != sid:
                raise AssembledError(f"ID {sid} model/address verification failed")

    def _verify_six(self, ids):
        if len(ids) != EXPECTED_COUNT or len(set(ids)) != EXPECTED_COUNT:
            raise AssembledError(
                "Exactly six distinct STS3215 motors must be found before joint mapping"
            )
        for sid in ids:
            self._verify_id(sid)
        return sorted(ids)

    def _reboot(self):
        self._write(254, reg.TORQUE_ENABLE, 0)
        self._sleep(0.03)
        with self.io_lock:
            self._check()
            # A timeout cannot prove whether the reboot packet was accepted.
            # Some firmware may restore torque while booting; retain a bounded
            # post-boot OFF obligation even if Stop arrives during this send.
            self.reboot_cleanup_at = time.monotonic() + 4.0
            self._publish(reboot_cleanup={"pending": True, "window_s": 4.0})
            try:
                self._send(packet(254, 0x08))
            finally:
                # Count from the last possible transmitted byte, including an
                # uncertain write timeout, rather than from journal I/O start.
                self.reboot_cleanup_at = time.monotonic() + 4.0

    def _wait_for_boot_cleanup(self, seconds):
        # Stop deliberately waits for this bounded boot window. It must not
        # close the adapter while a reboot can still restore firmware torque.
        time.sleep(max(0, seconds))

    def _boot_verified_off(self):
        self.reboot_cleanup_at = None
        self._publish(reboot_cleanup={"pending": False, "verified_off": True})

    def _measure_edge(self, sid):
        if self.unverified_models and not self.verified_so101:
            raise AssembledError(
                "Colliding motors could not identify their model. Confirm this is a "
                "six-STS3215 SO101 before experimental reboots."
            )
        edges = []
        with self.io_lock:
            original_timeout = getattr(self.bus, "timeout", None)
            if original_timeout is not None:
                self.bus.timeout = 0.001
            try:
                for _ in range(3):
                    self._reboot()
                    start = time.monotonic()
                    self._sleep(0.70)
                    while time.monotonic() - start < 4.0:
                        # An expected position reply is eight bytes. Reading64
                        # with normal6ms timeout would date the end of a timeout,
                        # shifting the narrow sweep past the actual wake window.
                        response, _ = self._read_response(
                            sid, reg.PRESENT_POSITION, 2, wake_poll=True
                        )
                        if response:
                            edges.append(time.monotonic() - start)
                            self._write(254, reg.TORQUE_ENABLE, 0)
                            break
                    else:
                        raise AssembledError(
                            "Cannot measure this firmware/adapter reboot wake edge; no ID "
                            "sweep was sent"
                        )
                    self._sleep(0.25)
                    self._write(254, reg.TORQUE_ENABLE, 0)
            finally:
                if original_timeout is not None:
                    self.bus.timeout = original_timeout
        if max(edges) - min(edges) > 0.020:
            raise AssembledError(
                "Reboot wake timing is unstable on this host; no ID sweep was sent"
            )
        edge = min(edges)
        self._publish(
            wake_edge_ms=round(edge * 1000, 3),
            message="Measured reboot timing; preparing bounded ID separation",
        )
        return edge

    def _rename(self, old, new):
        """One durable rename transaction; Stop waits for its critical section.

        Cancellation is checked before every normal packet. If cancellation or a
        transport error arrives after an ID write, the only allowed cleanup is
        re-locking that ID before releasing io_lock; never a goal or torque-on.
        """
        self._publish(
            pending={"old": old, "new": new, "possible_ids": sorted({old, new}), "stage": "intent"}
        )
        with self.io_lock:
            self._check()
            if old != new:
                response, _ = self._read_response(new, reg.PRESENT_POSITION, 2)
                if response:
                    raise AssembledError(
                        f"Temporary destination ID {new} is occupied; no rename sent"
                    )
            unlocked = attempted_id = False
            try:
                self._write(old, reg.TORQUE_ENABLE, 0)
                self._sleep(0.02)
                self._write(old, reg.LOCK, 0)
                unlocked = True
                self._sleep(0.02)
                self._publish(
                    pending={
                        "old": old,
                        "new": new,
                        "possible_ids": sorted({old, new}),
                        "stage": "id-write-uncertain",
                    }
                )
                attempted_id = True
                self._write(old, reg.ID, new)
                self._sleep(0.02)
                self._write(new, reg.LOCK, 1, cleanup=True)
                unlocked = False
                self._sleep(0.03)
                self._verify_id(new)
                if self._read(new, reg.LOCK) != 1 or self._read(new, reg.TORQUE_ENABLE) != 0:
                    raise AssembledError("ID persistence/torque-off read-back failed")
            finally:
                if unlocked:
                    # A write timeout can happen after the servo accepted the
                    # new ID. Journal and lock both possible addresses; these
                    # targets were collision-free when the transaction began.
                    possible = [new, old] if attempted_id else [old]
                    cleanup_errors = []
                    for sid in dict.fromkeys(possible):
                        try:
                            self._write(sid, reg.LOCK, 1, cleanup=True)
                        except Exception as error:
                            cleanup_errors.append(str(error))
                    if cleanup_errors:
                        self._publish(
                            pending={
                                "old": old,
                                "new": new,
                                "possible_ids": sorted({old, new}),
                                "stage": "relock-uncertain",
                                "errors": cleanup_errors,
                            }
                        )
        ids = [new if sid == old else sid for sid in self.state["ids"]]
        self._publish(ids=sorted(set(ids)), pending=None, last_rename={"old": old, "new": new})

    def split(self):
        self.deadline = time.monotonic() + self.timeout_s
        try:
            self._open()
            self._publish(
                phase="splitting", error=None, message="Read-only scan of the assembled chain"
            )
            state = self._scan()
            if not state:
                raise AssembledError("No servo responses; check power, USB and the 1 Mbps bus")
            if len(state) > EXPECTED_COUNT:
                raise AssembledError(
                    "More than six addressed motors found; this setup supports one SO-101 "
                    "chain only"
                )
            if len(state) == EXPECTED_COUNT and all(kind == "single" for kind in state.values()):
                ids = self._verify_six(list(state))
            else:
                suspect = next((sid for sid, kind in state.items() if kind != "single"), min(state))
                edge = self._measure_edge(suspect)
                ids = None
                for number in range(1, MAX_ROUNDS + 1):
                    self._publish(
                        round=number,
                        message=f"Separating factory IDs: round {number}/{MAX_ROUNDS}",
                    )
                    suspect = next(
                        (sid for sid, kind in state.items() if kind != "single"), min(state)
                    )
                    fresh = [sid for sid in range(2, 90) if sid not in state][:84]
                    if not fresh:
                        raise AssembledError("No free temporary IDs remain for the bounded sweep")
                    burst = b"".join(
                        packet(suspect, 0x03, bytes([reg.LOCK, 0]))
                        + packet(suspect, 0x03, bytes([reg.ID, sid]))
                        for sid in fresh
                    )
                    self._publish(
                        pending={
                            "stage": "sweep-intent",
                            "source_id": suspect,
                            "possible_ids": fresh,
                            "round": number,
                        }
                    )
                    self._reboot()
                    start = time.monotonic()
                    self._sleep(max(0, edge - 0.006 - 0.002))
                    while time.monotonic() - start < edge - 0.006:
                        self._check()
                    self._send(burst)
                    self._sleep(0.15)
                    self._write(254, reg.TORQUE_ENABLE, 0)
                    state = self._scan()
                    if len(state) > EXPECTED_COUNT:
                        raise AssembledError(
                            "Unexpected motor inventory after splitting; inspect journal "
                            "before retrying"
                        )
                    # Re-lock every independently responding ID before rebooting
                    # again. Keep ambiguous/identical replies out of final acceptance.
                    for sid, kind in sorted(state.items()):
                        if kind == "single":
                            self._rename(sid, sid)
                    if len(state) == EXPECTED_COUNT and all(
                        kind == "single" for kind in state.values()
                    ):
                        ids = self._verify_six(list(state))
                        break
                if ids is None:
                    raise AssembledError(
                        "This host/firmware did not separate all six motors within the "
                        "bounded attempts; partial IDs are journaled"
                    )
            for sid in ids:
                self._write(sid, reg.TORQUE_ENABLE, 0)
                if self._read(sid, reg.TORQUE_ENABLE) != 0:
                    raise AssembledError(
                        "Motor torque-off read-back failed; manual mapping is blocked"
                    )
            self._boot_verified_off()
            self._publish(
                phase="ready",
                ids=ids,
                mapping={},
                pending=None,
                message=(
                    "Six distinct motors verified; identify each joint by moving it gently by hand"
                ),
            )
            return ids
        except Exception as error:
            self._publish(
                phase="cancelled" if isinstance(error, AssembledCancelled) else "error",
                error=str(error),
                message=str(error),
            )
            raise
        finally:
            self.deadline = None

    def positions(self):
        self._check()
        ids = self.state["ids"]
        if (
            self.state["phase"] not in ("ready", "observing", "mapped", "complete")
            or len(ids) != EXPECTED_COUNT
        ):
            raise AssembledError("Verify six distinct motors before observing manual movement")
        positions = {sid: self._read(sid, reg.PRESENT_POSITION, 2) for sid in ids}
        if any(type(value) is not int or not 0 <= value <= 4095 for value in positions.values()):
            raise AssembledError(
                "A joint position is missing or invalid; stop and scan the chain again"
            )
        self._publish(journal=False, positions=positions)
        return positions

    def apply(self, mapping):
        self.deadline = time.monotonic() + self.timeout_s
        try:
            self._check()
            if (
                set(mapping) != set(JOINTS)
                or any(type(value) is not int for value in mapping.values())
                or len(set(mapping.values())) != EXPECTED_COUNT
                or set(mapping.values()) != set(self.state["ids"])
            ):
                raise AssembledError(
                    "Map each of the six named joints to a different discovered motor before saving"
                )
            actual = self._scan()
            if set(actual) != set(mapping.values()) or any(
                kind != "single" for kind in actual.values()
            ):
                raise AssembledError(
                    "Motor inventory changed after mapping; no IDs saved. Scan and "
                    "identify the six joints again."
                )
            self._verify_six(list(mapping.values()))
            self._publish(
                phase="committing",
                mapping=dict(mapping),
                message="Saving reviewed joint identities through collision-free temporary IDs",
            )
            occupied = set(actual) | set(IDS.values())
            temporary = [sid for sid in range(100, 254) if sid not in occupied][:EXPECTED_COUNT]
            if len(temporary) != EXPECTED_COUNT:
                raise AssembledError("No collision-free temporary ID space available")
            temp_map = dict(zip(JOINTS, temporary, strict=True))
            for joint in JOINTS:
                self._rename(mapping[joint], temp_map[joint])
            for joint in JOINTS:
                self._rename(temp_map[joint], IDS[joint])
            self._verify_six(list(IDS.values()))
            self._publish(message="Rebooting once to verify the six saved IDs persist")
            self._reboot()
            self._sleep(2.0)
            self._write(254, reg.TORQUE_ENABLE, 0)
            # Some firmware boots more slowly than the commonly reported800ms.
            end = time.monotonic() + 4.0
            while time.monotonic() < end:
                if all(self._read(sid, reg.ID) == sid for sid in IDS.values()):
                    break
                self._sleep(0.1)
            else:
                raise AssembledError(
                    "Saved IDs did not survive reboot; no success is claimed. Inspect the "
                    "setup journal."
                )
            final = self._verify_six(list(IDS.values()))
            for sid in final:
                if self._read(sid, reg.LOCK) != 1 or self._read(sid, reg.TORQUE_ENABLE) != 0:
                    raise AssembledError("Post-reboot lock/torque-off verification failed")
            self._boot_verified_off()
            self._publish(
                phase="complete",
                ids=final,
                mapping={joint: IDS[joint] for joint in JOINTS},
                pending=None,
                message=(
                    "IDs1\u20136 verified after reboot; proceed to calibration with all cables "
                    "connected"
                ),
            )
            return final
        except Exception as error:
            self._publish(
                phase="cancelled" if isinstance(error, AssembledCancelled) else "error",
                error=str(error),
                message=str(error),
            )
            raise
        finally:
            self.deadline = None

    def close(self):
        """Close serial after bounded post-boot torque-OFF cleanup if needed.

        No writes occur unless this engine previously attempted a reboot that
        has not yet been followed by six individual torque-off readbacks. Stop
        waits under io_lock for at most the remaining four-second boot window
        and bounded serial send; it permits only this essential TorqueOFF packet.
        Cleanup failure is journaled and raised after releasing the serial port.
        """
        with self.io_lock:
            if self.bus is None:
                return
            cleanup_error = None
            try:
                if self.reboot_cleanup_at is not None:
                    self._wait_for_boot_cleanup(self.reboot_cleanup_at - time.monotonic())
                    try:
                        self._write(254, reg.TORQUE_ENABLE, 0, cleanup=True)
                    except Exception as error:
                        cleanup_error = error
                    self._publish(
                        reboot_cleanup={
                            "pending": cleanup_error is not None,
                            "attempted": True,
                            "verified_off": False,
                            "error": str(cleanup_error) if cleanup_error else None,
                        }
                    )
                    if cleanup_error is None:
                        self.reboot_cleanup_at = None
            finally:
                try:
                    self.bus.close()
                finally:
                    self.bus = None
            if cleanup_error is not None:
                raise AssembledError(
                    "Post-reboot torque-off cleanup failed; keep supporting the arm and "
                    f"check power/USB: {cleanup_error}"
                ) from cleanup_error
