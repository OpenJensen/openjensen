# Devices in Robot Lab

Initial setup has three tabs: **Devices**, **Teleoperation**, and **Record**.

Devices lists connected USB controllers and saved devices, including disconnected ones. A tile shows its saved name (or USB address), connection state, motor-ID setup, and calibration status. Opening a new device asks for its default leader/follower role. Its detail page contains Overview, Motor IDs, and Calibration, including calibration export/import. The name and default role are editable there. Discovery and metadata edits do not command motors.

Names belong to device identities, rather than the two session slots. A unique USB serial number keeps an identity when the operating system assigns a different port; controllers without unique serial numbers use their port identity. Existing calibration profiles are recovered as saved devices. Profiles without a recorded port can be assigned a connected USB connection in their settings. Renaming a device does not rename or discard its calibration file. Changing the default role retains the same calibration and setup revision.

Teleoperation and recording choose actual saved devices for the session’s follower and optional leader slots. Their default roles provide initial suggestions; either configured device can be assigned either session role. A session cannot use the same USB connection twice. Changing selections never starts motion, and selections are locked while control or recording is active. Existing motor checks, calibration comparisons, torque controls, watchdogs, and explicit motion acknowledgements remain in force.

Device records are stored by the robot connection service in its private data directory as `devices.json`. Install the updated companion together with the application using the normal `deploy/install_robot_lab.py` installer. Old calibration JSON and setup journals remain in place. The connection banner reports USB devices; it no longer describes the computer as a robot connection.
