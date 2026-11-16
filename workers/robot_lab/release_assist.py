"""Turn a policy's request to open a gripper that is holding something into a full release.

A position-controlled gripper that closes on an object stops at the object's width: the
command says "closed" (near 0%) while the jaw rests at, say, 6%. Any command between the two
moves nothing. Policies learn the release as a short, timid bump in the gripper command, so
their attempt to let go usually stays inside that gap and the object is never dropped.

While the jaw is held open by an object, a command above where the jaw rests can only mean
"open"; this opens the gripper fully for a moment instead.
"""

# Holding: a closing command, with the jaw this far above it, for this long.
HOLD_MAX_COMMAND = 12.0  # percent
HOLD_GAP = 2.5  # percent
HOLD_SECONDS = 0.4
# The jaw closing this far below where it was held means the object slipped out.
SLIP = 2.0  # percent
# Releasing: a command this far above the held jaw opens fully, for this long.
RELEASE_MARGIN = 0.5  # percent
RELEASE_OPEN = 30.0  # percent; recorded duck releases open to 16-31%
RELEASE_SECONDS = 1.0


class ReleaseAssist:
    def __init__(self):
        self.reset()

    def reset(self):
        self.blocked_since = None
        self.held_jaw = None
        self.release_until = 0.0
        self.releases = 0

    @property
    def holding(self):
        return self.held_jaw is not None

    def gripper(self, command, measured, now):
        """The gripper target to send for the policy's command and the measured jaw."""
        if now < self.release_until:
            return max(command, RELEASE_OPEN)
        if measured is None:
            return command
        if self.held_jaw is not None:
            if command > self.held_jaw + RELEASE_MARGIN:
                self.releases += 1
                self.release_until = now + RELEASE_SECONDS
                self.held_jaw = self.blocked_since = None
                return max(command, RELEASE_OPEN)
            if measured < self.held_jaw - SLIP:
                self.held_jaw = self.blocked_since = None
            else:
                # Follow the jaw slowly: it settles as the grip tightens.
                self.held_jaw += 0.2 * (measured - self.held_jaw)
            return command
        if command <= HOLD_MAX_COMMAND and measured - command >= HOLD_GAP:
            if self.blocked_since is None:
                self.blocked_since = now
            elif now - self.blocked_since >= HOLD_SECONDS:
                self.held_jaw = measured
        else:
            self.blocked_since = None
        return command
