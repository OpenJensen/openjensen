# SO-101 Robot Lab

Robot Lab adds the physical-robot workflow to OpenJensen: local USB devices and camera preview, initial joint IDs, staged calibration, bounded manual control or leader-arm teleoperation, episode recording, and project dataset import. It uses a small local companion on the computer connected to the arm. GPU training workers remain separate.

## Connect a Mac

Install `uv` and use Python 3.14.7. From a checkout run:

```sh
python3 deploy/macos/install_robot_lab.py --source workers/robot_lab --ssh-host xbox-360
```

The installer creates an isolated runtime under `~/.local/share/jensen/robot-lab`, private data under `~/Library/Application Support/OpenJensen/robot-lab`, and two user LaunchAgents: `com.openjensen.robot-lab` and `com.openjensen.robot-lab.tunnel`. The companion listens only on 127.0.0.1:18101; the private SSH tunnel exposes it only on Xbox loopback and opens the private workspace at http://127.0.0.1:18096/. The operator must already have authorized SSH access to that host.

An existing Robot Lab profile environment can be reused with `--runtime workers/robot_lab/.venv`. An existing compatible PyArrow environment can be reused with `--arrow-site-packages /absolute/path/to/site-packages`. With cached camera/servo dependencies, `--offline` prevents downloads. The installation receipt records this runtime dependency; preserve that PyArrow environment while the companion uses it. Installations require at least 10 GiB free disk space before adding dependencies. This hardware runtime does not download model weights or install a GPU training stack.

On the application host, write a private owned mode-0600 JSON file at `<workspace>/robot-lab-connection.json`:

```json
{"url":"http://127.0.0.1:18101","token":"COPY_THE_PRIVATE_COMPANION_TOKEN","label":"My Mac"}
```

The token is read from the companion's private `companion-token` file; it never goes to the browser. An operator can override the connection-file path with `JENSEN_ROBOT_LAB_CONFIG`. Only loopback HTTP endpoints are accepted. Do not expose either service port publicly. Public gateways deny Robot Lab device, camera, control and import APIs, even for signed-in viewers.

The SO-101 controller USB port and camera must be visible to the Mac. If a USB-A-host sharing hub came with an A-to-A cable, keep its supplied cable and attach through a USB-A host adapter or a working monitor hub. Use the hub's labeled power input and power specification; Waveshare USB 2.0 and USB 3.2 versions have different DC requirements. Camera access may need approval in macOS Privacy & Security. Only the selected camera is captured, and the companion shares its frames between preview and recording.

## Guided workflow

Choose a project and open **Robot Lab**. **Connect** lists local serial adapters and cameras and probes motors without enabling torque. A factory-ID response cannot prove that only one physical motor is attached.

**Motor IDs** requires an explicit isolation acknowledgment for each motor. Disconnect motor power before changing servo cables, connect just the named motor to the board, restore its correct motor power, and assign the ID shown. The guide follows gripper 6, wrist roll 5, wrist flex 4, elbow 3, shoulder lift 2, shoulder pan 1. New motors sharing ID 1 cannot be distinguished reliably while daisy-chained. Do not assign IDs to the complete factory chain.

**Calibrate** first verifies all six IDs and disables torque. Support the arm, capture the middle pose, move the joints gently through their usable ranges, then save. Saved calibration uses the official LeRobot JSON structure and namespace. Range validation rejects incomplete or invalid measurements; cancel preserves the previous calibration. The guide does not command automatic sweeps.

**Teleoperate** requires a saved calibration. Manual jogs are small and bounded; leader mode requires a separate calibrated leader controller. Hand-guide mode leaves torque disabled and can record passive demonstrations. Enabling motion is an explicit UI action, and targets start at observed positions. The session stops on serial errors, an expired browser heartbeat or **Stop**. Disabling torque can let the arm fall; support it and keep the motor power switch accessible.

**Record** captures calibrated states, actual commanded actions and the selected camera into a local LeRobot v3 dataset. Enter a task and dataset name, start an episode, then save or discard that take. Finish before importing the dataset into the selected project. No automatic Hugging Face upload occurs. Capture errors remain visible; partial episodes are not reported as successful demonstrations. Monitor disk space during long captures.

## Operations and validation

Logs and the token stay under the companion installation directory. Restart with `launchctl kickstart -k gui/$(id -u)/com.openjensen.robot-lab`; use the same command with `com.openjensen.robot-lab.tunnel` for the SSH connection. Stop a service with `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/<label>.plist`. Stopping or uninstalling the runtime must not delete captured datasets or calibrations.

Companion API details are in [the protocol](../workers/robot_lab/API.md). Automated tests use injected fake buses and cameras; real-arm motion, ID changes and calibration require a connected, prepared arm and are separate from contract/browser test success.
