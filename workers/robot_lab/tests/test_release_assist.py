import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from release_assist import HOLD_SECONDS, RELEASE_OPEN, RELEASE_SECONDS, ReleaseAssist  # noqa: E402


def hold(assist, jaw=6.0, command=0.5, start=0.0):
    """Close on an object: the command says closed while the jaw rests at the object's width."""
    t = start
    while t <= start + HOLD_SECONDS + 0.1:
        assert assist.gripper(command, jaw, t) == command
        t += 0.05
    return t


class ReleaseAssistTests(unittest.TestCase):
    def test_a_command_above_the_held_jaw_opens_fully_for_a_moment(self):
        """Recorded on the arm: holding the duck at 6%, the policy asked for 6.9%."""
        assist = ReleaseAssist()
        t = hold(assist, jaw=5.9)
        self.assertTrue(assist.holding)
        self.assertEqual(assist.gripper(6.9, 5.9, t), RELEASE_OPEN)
        self.assertEqual(assist.gripper(0.5, 8.0, t + RELEASE_SECONDS / 2), RELEASE_OPEN)
        self.assertEqual(assist.gripper(0.5, 25.0, t + RELEASE_SECONDS + 0.01), 0.5)
        self.assertEqual(assist.releases, 1)
        self.assertFalse(assist.holding)

    def test_openings_inside_the_gap_and_the_closing_itself_do_not_release(self):
        assist = ReleaseAssist()
        t = hold(assist, jaw=6.6)
        for command in (0.5, 1.4, 3.8, 4.7, 6.1):  # all at or below where the jaw rests
            self.assertEqual(assist.gripper(command, 6.6, t), command)
            t += 0.05
        self.assertEqual(assist.releases, 0)
        self.assertTrue(assist.holding)

    def test_no_release_without_an_object_between_the_jaws(self):
        assist = ReleaseAssist()
        # Closing on nothing: the jaw follows the command, so nothing holds it open.
        for i in range(40):
            self.assertEqual(assist.gripper(0.5, 1.5, i * 0.05), 0.5)
        self.assertFalse(assist.holding)
        self.assertEqual(assist.gripper(20.0, 1.5, 2.1), 20.0)
        # A brief stall while the jaw is still moving is not a hold.
        self.assertEqual(assist.gripper(0.5, 9.0, 3.0), 0.5)
        self.assertEqual(assist.gripper(10.0, 9.0, 3.1), 10.0)
        self.assertEqual(assist.releases, 0)

    def test_a_slipped_object_ends_the_hold(self):
        assist = ReleaseAssist()
        t = hold(assist, jaw=6.0)
        assist.gripper(0.5, 1.0, t)  # the jaw closed: the duck slipped out
        self.assertFalse(assist.holding)
        self.assertEqual(assist.gripper(8.0, 1.0, t + 0.05), 8.0)

    def test_a_larger_opening_than_the_assist_is_kept(self):
        assist = ReleaseAssist()
        t = hold(assist, jaw=6.0)
        self.assertEqual(assist.gripper(45.0, 6.0, t), 45.0)
        self.assertEqual(assist.releases, 1)


if __name__ == "__main__":
    unittest.main()
