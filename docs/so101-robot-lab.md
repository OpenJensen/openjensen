# SO-101 Robot Lab

Robot Lab adds the physical-robot workflow to OpenJensen: local USB devices and camera preview, initial joint IDs, staged calibration, bounded manual control or leader-arm teleoperation, episode recording, and project dataset import. It uses a small local companion on the computer connected to the arm. GPU training workers remain separate.

## Connect macOS or Linux

Install `uv` and use Python 3.14.7. From a checkout run:

```sh
# macOS
python3 deploy/macos/install_robot_lab.py --source workers/robot_lab --ssh-host application-server
# Linux
python3 deploy/linux/install_robot_lab.py --source workers/robot_lab --ssh-host application-server
```

On macOS, the installer creates an isolated runtime under `~/.local/share/jensen/robot-lab`, private data under `~/Library/Application Support/OpenJensen/robot-lab`, and two user LaunchAgents: `com.openjensen.robot-lab` and `com.openjensen.robot-lab.tunnel`. The companion listens only on 127.0.0.1:18101; the private SSH tunnel exposes it only on the application server loopback and opens the private workspace at http://127.0.0.1:18096/. The operator must already have authorized SSH access to that host.

On Linux, the same isolated hardware runtime uses systemd user services and private capture data under `~/.local/share/OpenJensen/robot-lab` (or `$XDG_DATA_HOME/OpenJensen/robot-lab`). See [Linux installation and operations](../deploy/linux/README.md) for device permissions, headless setup and service commands. The companion supports USB serial adapters and V4L2 cameras; it excludes metadata-only camera nodes without starting capture. Stop the previous computer's SSH tunnel before switching the bridge to another computer.

An existing Robot Lab profile environment can be reused with `--runtime workers/robot_lab/.venv`. An existing compatible PyArrow environment can be reused with `--arrow-site-packages /absolute/path/to/site-packages`. With cached camera/servo dependencies, `--offline` prevents downloads. The installation receipt records this runtime dependency; preserve that PyArrow environment while the companion uses it. Installations require at least 10 GiB free disk space before adding dependencies. This hardware runtime does not download model weights or install a GPU training stack.

On the application host, write a private owned mode-0600 JSON file at `<workspace>/robot-lab-connection.json`:

```json
{"url":"http://127.0.0.1:18101","token":"COPY_THE_PRIVATE_COMPANION_TOKEN","label":"Robot computer"}
```

The token is read from the companion's private `companion-token` file; it never goes to the browser. An operator can override the connection-file path with `JENSEN_ROBOT_LAB_CONFIG`. Only loopback HTTP endpoints are accepted. Do not expose either service port publicly. Public gateways deny Robot Lab device, camera, control and import APIs, even for signed-in viewers.

The SO-101 controller USB port and camera must be visible to the computer running the companion. If a USB-A-host sharing hub came with an A-to-A cable, keep its supplied cable and attach through a USB-A host adapter or a working monitor hub. Use the hub's labeled power input and power specification; Waveshare USB 2.0 and USB 3.2 versions have different DC requirements. Camera access may need approval in macOS Privacy & Security. Only the selected camera is captured, and the companion shares its frames between preview and recording.

## Guided workflow

Choose a project and open **Robot Lab**. **Connect** lists local serial adapters and cameras and probes motors without enabling torque. A factory-ID response cannot prove that only one physical motor is attached.

**Motor IDs** opens an interactive view of the assembled SO-101. Select a numbered joint to highlight it in the SO-101 CAD model. Use the motion preview or position slider to inspect its axis; selection itself is static. The model uses the repository’s URDF and meshes and does not represent live encoder positions or command the arm. A static view remains available with reduced motion.

The **assembled-arm setup** is experimental. It adapts [Enigma's STS3215 research](https://www.enigma.inc/posts/sts3215-servo-hacking) and [MIT-licensed servotools](https://github.com/Enigma-Incorporated/servotools/tree/1c65c57587f91085cd32fca3743f3e6f464cfe35): first separate duplicate factory IDs using measured reboot timing, then identify each physical joint from manual encoder movement. Keep every servo cable connected. Support the arm before starting: rebooting and disabling torque can let it fall. Starting requires an explicit acknowledgement because it reboots servos and changes persistent IDs, even though it never commands powered movement.

After six separately responding STS3215 motors are found, choose **Move by hand** to select a joint, move it gently, and confirm the detected address. Alternatively choose **Wiggle a motor**, select a temporary address, support the arm with all joints clear of their stops, and explicitly request its brief low-torque motion. Select the joint you actually saw move and confirm that match. Wiggle mode releases torque afterward and requires measurable encoder movement; it cannot operate on duplicate or unverified addresses. Repeat for all six joints, review the mapping, then save. The app renumbers through unused temporary addresses and verifies the canonical mapping: shoulder pan 1, shoulder lift 2, elbow 3, wrist flex 4, wrist roll 5, gripper 6. Previous calibration files are preserved but require fresh calibration after this setup. Stop cancels new work, finishes any in-flight persistence cleanup, and retains a recovery record of partial addressing. Each separation round rescans saved addresses after reboot so RAM-only sweep addresses are never reused as reboot targets. Clearly early/late sweep outcomes adjust the bounded timing window; ambiguity never authorizes powered identification.

This is a timing workaround, not a documented unique-serial addressing feature. Simulator tests cannot establish compatibility with every servo firmware, USB adapter or host. Unsupported timing, ambiguous movement, missing motors and verification errors leave setup incomplete. The standard isolated-motor method remains in the advanced section; its ID assignment button still requires physical isolation and is never used on the fully connected chain.

**Calibrate** first verifies all six IDs and disables torque. Support the arm, capture the middle pose, move the joints gently through their usable ranges, then save. Saved calibration uses the official LeRobot JSON structure and namespace. Range validation rejects incomplete or invalid measurements; cancel preserves the previous calibration. The guide does not command automatic sweeps.

**Teleoperate** requires a saved calibration. Joint control uses LeRobot SO101 configuration and direct calibrated position targets; leader mode requires a separate calibrated leader controller. Hand-guide mode leaves torque disabled and can record passive demonstrations. Enabling motion is an explicit UI action. Manual control initially holds the current pose; in leader mode the follower immediately tracks the leader using full calibrated targets. Sliders, steps and movement to the recorded middle send full requested positions and may move faster than earlier versions. Keep the entire path clear. The session stops on serial errors, an expired browser heartbeat or **Stop**. Disabling torque can let the arm fall; support it and keep the motor power switch accessible.

**Record** captures calibrated states, actual commanded actions and the selected camera into a local LeRobot v3 dataset. Enter a task and dataset name, start an episode, then save or discard that take. Finish before importing the dataset into the selected project. No automatic Hugging Face upload occurs. Capture errors remain visible; partial episodes are not reported as successful demonstrations. If a recording has no saved episodes, **Close unsaved recording** stops capture and preserves its unsaved files. Create a new dataset with a different name to retry at a lower frame rate. Closed recordings cannot be imported as finalized datasets. Monitor disk space during long captures.

## Operations and validation

The token stays under the companion installation directory. Linux service logs use `journalctl --user -u openjensen-robot-lab.service`; restart with `systemctl --user restart openjensen-robot-lab.service`. macOS logs stay under the installation directory. Restart with `launchctl kickstart -k gui/$(id -u)/com.openjensen.robot-lab`; use the same command with `com.openjensen.robot-lab.tunnel` for the SSH connection. Stop a service with `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/<label>.plist`. Stopping or uninstalling the runtime must not delete captured datasets or calibrations.

Companion API details are in [the protocol](../workers/robot_lab/API.md). Automated tests use injected fake buses and cameras; real-arm motion, ID changes and calibration require a connected, prepared arm and are separate from contract/browser test success.


## LeRobot control profile

The companion reuses the Apache-2.0-licensed LeRobot v0.4.4 SO follower setup and
position-action methods. It uses the stock acceleration/PID settings and batched
position packets, with no relative-target clipping. The former custom slow
trajectories, speed cap and five-degree lag abort are removed. Gripper protection,
calibration limits, explicit enable and the control heartbeat remain in place.
The status response identifies the active profile; an older companion is labeled
as an earlier profile in the control screen.

Normal control also restores the volatile torque ceiling from the saved motor
maximum, so an interrupted low-torque identification cannot silently leave the
arm limited to 10%. This follows the manufacturer's [power-on register behavior](https://docs.waveshare.com/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation).
Physical motor torque has not been measured by the software checks.
If a shoulder or elbow still needs support, verify the actual servo variant and
its matching motor power supply, cable voltage drop and joint binding. Do not
choose a higher supply voltage without checking the motor variant. Hardware
acceptance requires a connected arm with the operator present.
