# Transfer SO-101 calibration

Calibration normally survives application restarts and motor power cycles. Saving a calibration writes motor offsets and limits, then saves a matching named profile on the robot computer. If another computer recalibrates that arm, bring its new profile to your other installation.

In **Initial setup → Calibrate**, select the leader or follower and its profile name.

- **Export calibration** downloads that saved profile as a portable JSON file. Export does not read or change motor settings.
- **Import calibration** accepts an OpenJensen export or a standard LeRobot calibration JSON file. Review its joint values and destination name, then choose **Save imported calibration**. A profile from another name can be imported for the same physical arm. OpenJensen exports retain the arm role and cannot be silently imported as the other role.
- Replacing an existing profile requires an explicit checkbox and retains the exact previous file as a timestamped backup beside it.

Import saves only the computer-side profile. It never writes motor registers or enables torque. With an arm connected, all six motor settings must match before the import is saved. A mismatch leaves the previous profile untouched. Without an arm, the file can be imported for later use, but motion still checks the connected motors before enabling control. A setup invalidated by motor ID changes still requires connected verification.

Transfer files between installations controlling the same arm. Importing a file for a different physical arm does not calibrate that arm. Connection credentials, device paths, camera settings and datasets are excluded from exports.
