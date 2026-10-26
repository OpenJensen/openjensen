# SO-101 Robot Lab

Robot Lab sets up and drives an SO-101 arm from the browser: find its motors, assign their IDs, calibrate, teleoperate and record LeRobot v3 datasets. A small companion runs on the computer the arm is plugged into and reaches the OpenJensen server through a private SSH tunnel. USB devices and recordings stay on that computer until you import a dataset. GPU training workers are separate.

## What you need

- An SO-101 follower arm with Feetech STS3215 servos, its controller board and motor power supply. A second, calibrated arm if you want leader-arm teleoperation.
- A USB camera, if you want to record video.
- A macOS or Linux computer for the arm, with Python 3.14.7, [uv](https://docs.astral.sh/uv/), SSH access to the OpenJensen server, and at least 10 GiB of free disk.

## 1. Install the companion

From a checkout on the computer with the arm:

```sh
# macOS
python3 deploy/macos/install_robot_lab.py --source workers/robot_lab --ssh-host application-server
# Linux
python3 deploy/linux/install_robot_lab.py --source workers/robot_lab --ssh-host application-server
```

Replace `application-server` with the SSH host alias of your OpenJensen server. The installer sets up an isolated runtime and two user services: the companion and its SSH tunnel. It does not install a GPU stack or download model weights.

| | macOS | Linux |
| --- | --- | --- |
| Runtime | `~/.local/share/jensen/robot-lab` | `~/.local/share/jensen/robot-lab` |
| Recordings and calibrations | `~/Library/Application Support/OpenJensen/robot-lab` | `~/.local/share/OpenJensen/robot-lab` (or `$XDG_DATA_HOME/OpenJensen/robot-lab`) |
| Services | LaunchAgents `com.openjensen.robot-lab` and `com.openjensen.robot-lab.tunnel` | systemd user services `openjensen-robot-lab` and `openjensen-robot-lab-tunnel` |

On Linux, see [Linux installation](../deploy/linux/README.md) for device permissions and headless setup. Useful installer options:

- `--runtime workers/robot_lab/.venv` reuses an existing Robot Lab environment.
- `--arrow-site-packages /absolute/path/to/site-packages` reuses a compatible PyArrow; keep that environment while the companion uses it.
- `--offline` installs from cached dependencies without downloading.

## 2. Connect it to the server

The companion listens only on `127.0.0.1:18101`, and the tunnel exposes it only on the server's loopback. On the server, create `<workspace>/robot-lab-connection.json`, owned by the service user with mode `0600`:

```json
{"url":"http://127.0.0.1:18101","token":"COPY_THE_PRIVATE_COMPANION_TOKEN","label":"Robot computer"}
```

Copy the token from the companion's private `companion-token` file; it never reaches the browser. `JENSEN_ROBOT_LAB_CONFIG` overrides the file location. Only loopback addresses are accepted.

- Never expose either port publicly. Public gateways refuse Robot Lab device, camera, control and import requests, even for signed-in viewers.
- To move the arm to another computer, stop the old computer's tunnel first.

## 3. Plug in the arm and camera

The controller's USB port and the camera must be visible to the computer running the companion.

- On macOS, approve camera access in **Privacy & Security** if asked.
- A USB-A sharing hub supplied with an A-to-A cable needs that cable, connected through a USB-A host adapter or a working monitor hub. Use the hub's labeled power input and rating; Waveshare USB 2.0 and USB 3.2 versions need different DC supplies.
- Linux supports USB serial adapters and V4L2 cameras, and skips metadata-only camera nodes.
- Only the selected camera is captured; preview and recording share its frames.

## 4. Set up and use the arm

Choose a project, open **Robot Lab** and work through the steps. The status bar shows the companion, USB and torque state, and **Stop arm** appears whenever motors may be powered.

**Connect.** Lists serial adapters and cameras and checks which motors respond, without enabling torque. A reply from a factory-default ID cannot prove only one motor is attached.

**Motor IDs.** Each joint needs its own ID: shoulder pan 1, shoulder lift 2, elbow 3, wrist flex 4, wrist roll 5, gripper 6. Select a joint in the 3D view to highlight it; the view is a model of the arm and never moves or reads the real one. There are two ways to assign IDs:

- *One motor at a time* (standard). Power off, connect only the motor being set, restore power, and assign its ID. Repeat for each joint.
- *Assembled arm* (experimental). Keep every cable connected and support the arm: this mode reboots servos and changes stored IDs, and torque can release. It separates duplicate factory IDs by reboot timing, then identifies each joint either by **Move by hand** (move the joint gently and confirm the detected address) or **Wiggle a motor** (with all joints clear, request a brief low-torque motion and confirm the joint that moved). Review the mapping and save. Existing calibrations are kept but must be redone.

**Calibrate.** Checks all six IDs and disables torque. Support the arm, capture the middle pose, then move each joint gently through its usable range and save. Incomplete or invalid ranges are rejected, and cancelling keeps the previous calibration. Calibrations use the LeRobot JSON format.

**Teleoperate.** Needs a saved calibration and the robot USB connected. Choose:

- *Joint control:* sliders and steps send calibrated position targets. The arm first holds its current pose.
- *Hand guiding:* torque stays off so you can move the arm by hand and record passive demonstrations.
- *Leader → follower:* a second calibrated arm drives the follower, which starts tracking immediately.

**Record.** Name the dataset and task, pick the camera and frame rate, then record episodes and save or discard each one. Recordings hold calibrated states, the actions actually commanded and the camera, in LeRobot v3 format. Finish the dataset to import it into the project; nothing uploads to Hugging Face automatically. An unsaved recording can be closed with **Close unsaved recording**, which keeps its files but cannot be imported. Watch disk space during long sessions.

## Safety

- Support the arm whenever torque may be released: it can fall.
- Keep **Stop arm** and the motor power switch within reach, and the arm's whole path clear.
- Moves are sent as full position targets and can be fast.
- Control stops on a serial error, on **Stop**, or when the browser's heartbeat expires (for example, a closed or hidden tab).
- Do not raise the supply voltage without confirming your servo variant. If a shoulder or elbow cannot hold the arm, check the servo variant, power supply, cable voltage drop and joint binding.

## Run the companion

| | Linux | macOS |
| --- | --- | --- |
| Logs | `journalctl --user -u openjensen-robot-lab.service` | In the runtime directory |
| Restart | `systemctl --user restart openjensen-robot-lab.service` (or `openjensen-robot-lab-tunnel.service`) | `launchctl kickstart -k gui/$(id -u)/com.openjensen.robot-lab` (or `com.openjensen.robot-lab.tunnel`) |
| Stop | `systemctl --user stop openjensen-robot-lab-tunnel.service openjensen-robot-lab.service` | `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/<label>.plist` |

Stopping or uninstalling the companion never deletes recordings or calibrations. The token stays in the companion's installation directory.

## Technical notes

- **Control.** The companion reuses the Apache-2.0 [LeRobot](https://github.com/huggingface/lerobot) v0.4.4 SO follower setup and position-action methods, with LeRobot's acceleration and PID settings and no relative-target clipping. Gripper protection, calibration limits, explicit enable and the heartbeat stay in place. Normal control restores the servos' torque limit from their saved maximum, matching their [power-on behavior](https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation), so an interrupted low-torque identification cannot leave the arm weak. The status response names the active control profile.
- **Assembled-arm setup.** Adapted from [Enigma's STS3215 research](https://www.enigma.inc/posts/sts3215-servo-hacking) and the MIT-licensed [servotools](https://github.com/Enigma-Incorporated/servotools/tree/1c65c57587f91085cd32fca3743f3e6f464cfe35). It relies on reboot timing, not a unique serial number, so some servo firmware, USB adapters or hosts may not support it. Ambiguous results leave setup incomplete; **Stop** keeps a recovery record of partial addressing.
- **Testing.** Automated tests use simulated buses and cameras. Real motion, ID changes and calibration need a connected arm with an operator present.
- **Protocol.** The companion API is described in [API.md](../workers/robot_lab/API.md).

## Save and return to a rest position

In Teleoperate, start full-arm hand guiding or joint control. Position the arm in your
preferred closed or supported resting pose and choose **Save current position as rest**.
The pose is kept on the robot computer for that follower and calibration. Recalibration
requires saving it again. In powered full-arm control, choose **Return to rest** before
releasing the motors. The arm moves gradually, pauses leader tracking, then holds the
saved pose. Keep the page visible and the entire path clear. Use **Stop & release motors**
once the arm is supported. Immediate Stop and connection-loss safety still release torque.
