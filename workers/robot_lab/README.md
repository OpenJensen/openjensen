# OpenJensen Robot Lab companion

A private macOS/Linux service that gives the OpenJensen web app access to the SO-101 arms and cameras on this computer. The app reaches it through a private SSH loopback tunnel. For installation and the browser workflow, see [SO-101 Robot Lab](../../docs/so101-robot-lab.md).

Never expose this service, its token or a USB serial port on a public network.

## Run

The [macOS](../../deploy/macos/README.md) and [Linux](../../deploy/linux/README.md) installers set this up as a user service. To run it in the foreground:

```sh
/path/to/runtime/bin/python /path/to/robot_lab/server.py \
  --host 127.0.0.1 --port 18101 \
  --token-file /private/path/companion.token \
  --data-dir '/private/path/robot-lab'
```

- **Dependencies:** PySerial, the Feetech SDK, OpenCV and NumPy for hardware; Pillow and PyArrow for recording. No Torch or LeRobot training install is needed. Versions are in [requirements-hardware.txt](requirements-hardware.txt).
- **Token:** a regular private file (mode `0600`) of at least 32 characters.
- **`--data-dir`:** datasets and pending episode frames.
- **Calibrations:** LeRobot's standard `~/.cache/huggingface/lerobot/calibration/robots/so_follower/` or `teleoperators/so_leader/`, honoring `HF_LEROBOT_CALIBRATION`. A replaced calibration file gets a timestamped backup.

The request and response contracts are in [API.md](API.md). The workflow follows the [official SO-101 guide](https://huggingface.co/docs/lerobot/main/en/so101) and [LeRobot v0.4.4](https://github.com/huggingface/lerobot/tree/v0.4.4/src/lerobot).

## Motor IDs

- A read-only scan never changes registers or torque. Factory motors that share ID 1 **cannot be told apart by an ordinary USB scan**, even when replies look stable.
- **Assembled-arm setup** (experimental) keeps every motor cable connected. Measured reboot timing separates duplicate addresses, then the operator moves each highlighted joint by hand and confirms its encoder identity. It reboots servos and changes persistent IDs, but never sends goal positions or enables torque. Support the arm first. A partial setup keeps a recovery journal and requires fresh calibration before control. The method adapts Enigma's MIT-licensed servotools research; firmware and host timing still need testing on real hardware.
- **Isolated-motor setup** (the official method) is an advanced option and still requires physically isolating each motor.
- ID setup, calibration and enabling control each need an explicit, named confirmation in the app.

## Calibration

Calibration disables torque, captures the middle pose held by hand, then samples each joint's range as it is swept by hand. Wrist roll uses the full 0–4095 range. Encoder errors or wraps abort the session and try to restore the previous motor calibration.

## Control

Powered control reuses the LeRobot v0.4.4 SO follower configuration and `send_action` methods in [lerobot_control.py](lerobot_control.py) ([license](LEROBOT-LICENSE), [provenance](LEROBOT-PROVENANCE.json)), without needing Torch.

**Motor settings.** Return_Delay_Time 0, Maximum_Acceleration 254, Acceleration 254 and P/I/D gains 16/0/32. Goal_Velocity is reset to its default. Only the gripper gets LeRobot's 500/250/25 torque, current and overload protection; arm current and torque limits are never raised above their saved hardware settings. The volatile torque limit is restored from the saved maximum (gripper 500), as at servo power-on, which also clears an interrupted 10% identification limit. Settings are read back and must match before a selected motor is enabled.

**Motion.** Sliders, steps, moving to the middle pose and leader tracking send full calibrated position goals with the SDK's GroupSyncWrite, following LeRobot's default `max_relative_target=None`. Full targets can move fast and push hard when a loaded joint lags its target. Keep the whole path clear. Encoder lag alone does not stop motion and does not indicate damage.

**Always enforced.** Calibration matching, holding the current pose before enabling, the selected-motor scope, calibrated position limits, hardware and communication faults, and explicit Enable and Stop. Torque is re-enabled only after all settings and hold targets are verified. Hand guiding keeps torque off. Software cannot make up for a weak or wrong power supply, voltage drop, a tight joint or the wrong servo variant.

**Watchdog.** Torque is disabled when the browser heartbeat is lost for 2 seconds (10 seconds during calibration). Stop and shutdown address every motor and close the ports the companion owns. A software stop needs working USB and power: support the arm before releasing torque, and switch motor power off if communication fails. Previews and status polling never command motion.

## Cameras

Inventory never opens every camera; only an explicit preview or recording opens the selected one, and a recording's preview reuses its latest frame.

- **macOS:** inventory uses AVFoundation's device list through ffmpeg, with `system_profiler` as a fallback. Opening runs through a bounded main-thread dispatcher, independent of the serial watchdog; timed-out opens are cancelled and late handles released. Approve **Privacy & Security → Camera** for the companion's Python process if asked.
- **Linux:** V4L2 capture and USB serial adapters (`/dev/ttyACM*` or `/dev/ttyUSB*`). Inventory reads V4L2 capabilities without streaming and skips metadata-only nodes. Add your user to the device groups (usually `dialout` and `video` on Ubuntu and Debian) and log in again. The service runs as a systemd user service or in the foreground.

## Recording format

Datasets are written directly as LeRobot **v3.0** image datasets, checked against LeRobot 0.4.4 without importing Torch on this computer:

- Parquet frame tables, embedded lossless RGB PNG images, task and episode metadata, and normalization statistics.
- Images in `observation.images.front`; states and actions in `observation.state` and `action`, in official joint order, in degrees and gripper percent.
- Hand-guided demonstrations record the observed position as the action; manual and leader recordings keep the bounded targets actually sent.
- Capture runs on fixed deadlines and shows the measured FPS. It pauses on an excessively late frame or when the rate stays below 90% of the request; metadata keeps each episode's measured FPS.
- Image statistics use 8×8 RGB samples; global quantiles are estimated from per-episode quantiles.

Episodes are buffered on disk and must be saved or discarded before finalizing; Stop keeps pending frames for that choice. Each episode is limited to five minutes. Creating a dataset needs 1 GiB free, and capture stops before the last 512 MiB. Finalized datasets give the app's importer a sorted file manifest with sizes and SHA-256 checksums. Local files are never changed, and nothing uploads to Hugging Face automatically.

## Tests

Unit, HTTP and Parquet tests use simulated buses and synthetic images:

```sh
python -m unittest discover -s workers/robot_lab/tests -v
```

The format test loads two recorded episodes, camera tensors, actions, tasks and action timestamps with LeRobot 0.4.4. The lightweight hardware runtime skips it; run it in an existing LeRobot environment:

```sh
OPENJENSEN_VERIFY_LEROBOT=1 /path/to/lerobot-env/bin/python \
  -m unittest discover -s workers/robot_lab/tests -p test_dataset.py -v
```

Passing simulated tests do not establish that calibration and motion work on a real arm; that needs a correctly wired and configured arm.
