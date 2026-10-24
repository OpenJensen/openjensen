# OpenJensen Robot Lab companion

This private macOS/Linux service gives the OpenJensen web app access to the
computer's USB SO-101 arms and cameras. Hardware operation requires PySerial,
the Feetech SDK, OpenCV and NumPy. Recording additionally uses Pillow and
PyArrow; a full local Torch/LeRobot training installation is unnecessary.
The app forwards requests through a private SSH loopback bridge. Never expose
this service, its token, or a USB serial port on a public network.

```sh
/path/to/runtime/bin/python /path/to/robot_lab/server.py \
  --host 127.0.0.1 --port 18101 \
  --token-file /private/path/companion.token \
  --data-dir '/private/path/robot-lab'
```

The token must be a regular private file (0600), at least 32 characters long.
The [macOS installer](../../deploy/macos/README.md) and [Linux installer](../../deploy/linux/README.md) keep source/runtime/config separately. `--data-dir`
contains datasets and pending episode frames. Calibration JSON is saved under
LeRobot's standard `~/.cache/huggingface/lerobot/calibration/robots/so_follower/`
or `teleoperators/so_leader/`, respecting `HF_LEROBOT_CALIBRATION`.
Existing calibration files receive timestamped backups when replaced.

Use the app's Connect → Motor IDs → Calibrate → Teleoperate → Record flow.
Factory motors sharing ID 1 **cannot be distinguished by an ordinary USB scan**,
even when a reply looks stable. The main Motor IDs screen now includes an animated
joint picker and experimental assembled-arm setup: measured reboot timing separates
duplicate addresses, then the operator moves each highlighted joint by hand and
confirms its encoder identity. All motor cables remain connected. The method adapts
Enigma's MIT-licensed servotools research; firmware/host timing still requires
hardware acceptance. It reboots servos and changes persistent IDs, never goal
positions or torque-on commands. Support the arm before starting. Partial setup
retains a recovery journal and requires fresh calibration before control.
The official isolated-motor workflow remains an advanced option and retains its
physical-isolation gate. A read-only scan never changes registers or torque.
ID setup, calibration, and enabling control require explicit named UI confirmations.

Calibration first disables torque, captures the manually held middle pose, and
samples every joint's manually swept range. Wrist roll uses the official full
0–4095 range. Encoder errors/wraps abort the session, attempting to restore the
previous motor calibration. Motion requires all six STS3215 models at IDs 1–6
and a saved calibration matching EEPROM. Enabling motion first writes all six
current positions as targets, limits speed and acceleration, then enables torque.
Manual jogs are capped at 2 degrees (gripper 2 percentage points), rate limited,
and constrained to measured joint limits. Leader control requires a separately
connected, calibrated leader arm. Handguide mode leaves torque off.

The software watchdog disables torque after a 2-second browser heartbeat loss
(10 seconds during calibration). Stop/shutdown attempt every motor and close
owned ports. A software stop depends on working USB/power; support the arm
before releasing torque and use motor power off if communication fails.
No web preview or status polling commands motion.

macOS camera inventory uses AVFoundation's device listing through ffmpeg, with
system_profiler as an inventory fallback. Inventory never opens every camera.
Only explicit preview/record operations open the selected camera. AVFoundation
permission/open runs through a bounded main-thread dispatcher, independent of
the serial watchdog. A recording's preview reuses its latest frame. Camera
access may need macOS Privacy & Security → Camera approval for the companion's
Python process. Timed out queued opens are cancelled and late handles released.

Linux uses V4L2 camera capture and USB serial adapters (`/dev/ttyACM*` or
`/dev/ttyUSB*`). Inventory queries V4L2 capabilities without starting streaming
and excludes metadata-only nodes. Check your device groups (usually `dialout`
and `video` on Ubuntu/Debian), then log in again after group changes. The Linux
installer uses systemd user services; foreground operation is also supported.

Recording writes LeRobot **v3.0** image datasets directly: Parquet frame tables,
embedded lossless RGB PNG images, task/episode metadata, and normalization stats.
The format is verified with LeRobot 0.4.4 without importing Torch on the hardware computer.
Images use `observation.images.front`; states/actions use `observation.state`
and `action` in official joint-name order, in degrees and gripper percent.
Handguide demonstrations record observed position as action; manual/leader
recordings retain the bounded targets actually sent. Capture is paced against
fixed deadlines, displays measured FPS, and pauses on excessive single-frame
delay or sustained capture below 90% of the requested rate. Metadata retains
measured per-episode FPS. Image statistics use 8×8 RGB samples; global quantiles
are estimates aggregated from per-episode quantiles.

Episodes are buffered on disk and must be explicitly saved or discarded before
finalizing. Stop preserves pending frames for save/discard. The writer stops
before disk exhaustion (1 GiB required to create a dataset; 512 MiB reserve during
capture), with a five-minute episode limit. Finalized datasets expose a sorted
file manifest with sizes and SHA-256 checksums to the app's project importer.
Original local files remain intact. No Hugging Face upload is automatic.

See [API.md](API.md) for request/response contracts and
[requirements-hardware.txt](requirements-hardware.txt) for hardware runtime versions.
The workflow follows the
[official SO-101 guide](https://huggingface.co/docs/lerobot/main/en/so101) and
[LeRobot v0.4.4 source](https://github.com/huggingface/lerobot/tree/v0.4.4/src/lerobot).

## Verification

Safe unit/HTTP/Parquet tests use fake buses and synthetic images only:

```sh
python -m unittest discover -s workers/robot_lab/tests -v
```

The format proof loads two recorded episodes, camera tensors, actions, tasks and
action delta timestamps with official LeRobot 0.4.4. Run in an existing compatible
environment; the lightweight hardware runtime skips this one test explicitly:

```sh
OPENJENSEN_VERIFY_LEROBOT=1 /path/to/lerobot-env/bin/python \
  -m unittest discover -s workers/robot_lab/tests -p test_dataset.py -v
```

A passing fake-bus test does not establish physical calibration/motion acceptance.
That requires the user's correctly wired and configured arm.
