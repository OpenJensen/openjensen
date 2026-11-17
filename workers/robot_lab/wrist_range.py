"""A follower wrist-roll range, recorded by hand and enforced on every motor command.

Calibration treats wrist roll as a full-turn joint (0-4095 ticks), so nothing limits how far it
is commanded. On an arm whose cables or camera mount stop the wrist, a leader arm can turn it
into that stop: the leader has no such stop, and once its angle passes +/-180 degrees the
follower's target jumps to the opposite side. The range is recorded by turning the follower's
wrist to both ends with torque off, saved for the calibration it was measured in, and applied
where every goal position is written.
"""

import hashlib
import json

from dataset import atomic_json, safe_slug

# Stay this far inside the ends the wrist was turned to.
MARGIN = 3.0  # degrees
# A recording narrower than this is a mistake, not the wrist's range.
MIN_SPAN = 30.0  # degrees


def wrapped(angle):
    """The same angle in (-180, 180]."""
    angle = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if angle == -180.0 else angle


def full_turn_angle(raw):
    """Degrees of a wrist roll reading, as calibration (a full turn, 0-4095) will read it."""
    return (raw - 4095 / 2) * 360 / 4095


def calibration_digest(calibration):
    return hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()


class RangeRecorder:
    """Follow the wrist angle continuously while it is turned by hand, across +/-180 degrees."""

    def __init__(self, angle):
        self.start = angle
        self.last = angle
        self.offset = 0.0
        self.low = self.high = 0.0

    def sample(self, angle):
        self.offset += wrapped(angle - self.last)
        self.last = angle
        self.low = min(self.low, self.offset)
        self.high = max(self.high, self.offset)

    def snapshot(self):
        return {
            "now": round(self.last, 1),
            "low": round(self.start + self.low, 1),
            "high": round(self.start + self.high, 1),
            "span": round(self.high - self.low, 1),
        }

    def result(self):
        """The range to save, in the calibration's degrees, a margin inside both ends."""
        if self.high - self.low < MIN_SPAN:
            raise ValueError(
                f"Turn the wrist to both of its ends first (recorded {self.high - self.low:.0f}°)"
            )
        low, high = self.start + self.low + MARGIN, self.start + self.high - MARGIN
        # The calibration's zero is the pose the arm works in, so the range must contain it.
        # Recorded past +/-180 degrees, the same range may sit one turn away.
        for shift in (0.0, -360.0, 360.0):
            if low + shift <= 0.0 <= high + shift:
                low, high = low + shift, high + shift
                break
        else:
            raise ValueError("Turn the wrist through its working position while recording")
        # Commands are angles within one turn; the part of a wider range beyond +/-180 degrees
        # could not be told apart from the other side, so it is left out.
        return round(max(low, -180.0 + MARGIN), 1), round(min(high, 180.0 - MARGIN), 1)


def clamp(raw, last, low, high):
    """Hold a goal outside [low, high] at the end nearest where the joint was last sent.

    Nearest to the last goal, not to the new one: a leader turned past +/-180 degrees asks for
    the far side, and the wrist must stay at the end it reached, not swing round to the other.
    """
    if low <= raw <= high:
        return raw
    reference = raw if last is None else last
    return low if abs(reference - low) <= abs(reference - high) else high


class WristRanges:
    def __init__(self, data_dir):
        self.root = data_dir / "wrist-ranges"

    def path(self, robot_id):
        return self.root / (safe_slug(robot_id) + ".json")

    def load(self, robot_id, calibration):
        """The saved range in degrees, if it was recorded with this exact calibration."""
        path = self.path(robot_id)
        try:
            if path.is_symlink() or path.stat().st_size > 65536:
                return None
            saved = json.loads(path.read_text())
            if saved["calibration"] != calibration_digest(calibration):
                return None
            low, high = float(saved["min"]), float(saved["max"])
        except OSError, ValueError, KeyError, TypeError:
            return None
        if not -180.0 <= low < high <= 180.0 or high - low < MIN_SPAN - 2 * MARGIN:
            return None
        return low, high

    def save(self, robot_id, low, high, calibration):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_json(
            self.path(robot_id),
            {"calibration": calibration_digest(calibration), "min": low, "max": high},
        )
        return low, high

    def clear(self, robot_id):
        try:
            self.path(robot_id).unlink()
        except FileNotFoundError:
            pass
