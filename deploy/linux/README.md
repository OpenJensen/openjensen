# Linux Robot Lab component

Run this on the Linux computer physically connected to the SO-101 controller and camera.
The installer runs with system Python 3.10+ and creates the Python 3.14.7 hardware runtime using [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
python3 deploy/linux/install_robot_lab.py --source workers/robot_lab --ssh-host application-server
```

Run it as your normal user, in a session with a working `systemctl --user` manager. It installs no Torch, CUDA, simulator or model weights.

| What | Where |
| --- | --- |
| Runtime | `~/.local/share/jensen/robot-lab` |
| Recordings and recovery data | `${XDG_DATA_HOME:-~/.local/share}/OpenJensen/robot-lab` (change with `--data-dir`) |
| systemd user units | `${XDG_CONFIG_HOME:-~/.config}/systemd/user`: `openjensen-robot-lab.service` and `openjensen-robot-lab-tunnel.service` |

- `--runtime`, `--offline` and `--arrow-site-packages` reuse compatible existing dependencies, as on macOS.
- Reinstalling keeps the token and data, and stops only these two services before replacing code.

## Connect to the application server

The companion listens on `127.0.0.1:18101`. `--ssh-host` takes an SSH alias you can already log in to, and creates a private reverse tunnel to the application server plus a local forward of the workspace at [localhost:18096](http://127.0.0.1:18096/robot-lab/). Then create the [connection file](../../docs/so101-robot-lab.md#2-connect-it-to-the-server) on the application server; the companion token never reaches the browser.

- Only one companion can hold the server's tunnel port. Stop the other computer's tunnel before switching.
- If the app runs on this same computer, omit `--ssh-host`, write its loopback connection file and open the app at its usual address.

## USB and camera permissions

USB controllers normally appear as `/dev/ttyACM*` or `/dev/ttyUSB*`; cameras use
V4L2 `/dev/video*`. Check the actual device ownership with `ls -l` and your groups
with `id`. On Ubuntu/Debian the serial group is usually `dialout` and cameras
usually use `video`. If those are the groups owning your devices, an administrator
can grant access with:

```sh
sudo usermod -aG dialout,video "$USER"
```

Log out completely and log in again so both the session and systemd user manager
inherit the new groups. Use your distribution's actual device groups when different.
Do not run the companion as root or make devices world-writable. Close other serial
or camera applications if a device is busy. Inventory queries capabilities without
streaming; metadata-only V4L2 nodes are excluded. Only preview/record opens a capture.

In WSL, USB devices must first be attached to the distribution and available to
Linux; a service cannot discover USB devices exposed only to Windows. See Microsoft's
[USB connection guide](https://learn.microsoft.com/windows/wsl/connect-usb).

## Operations and headless use

```sh
systemctl --user status openjensen-robot-lab.service openjensen-robot-lab-tunnel.service
systemctl --user restart openjensen-robot-lab.service
journalctl --user -u openjensen-robot-lab.service -n 50
systemctl --user disable --now openjensen-robot-lab-tunnel.service openjensen-robot-lab.service
```

Services start with the user manager. For operation while logged out, an administrator
can enable user lingering with `sudo loginctl enable-linger "$USER"`; this is optional
and is not changed by the installer. For a first install without a user manager, use
`--no-start` to prepare the runtime and units, then run the installed source manually:

```sh
~/.local/share/jensen/robot-lab/runtime/bin/python \
  ~/.local/share/jensen/robot-lab/source/server.py \
  --token-file ~/.local/share/jensen/robot-lab/private/companion-token
```

The default Linux data directory is the same for managed and foreground operation.
Without systemd, maintain the private SSH tunnel yourself if the app is remote.
A reinstall requires a working user manager when managed units already exist, so
loaded code cannot be replaced underneath an existing companion. Never run two
companions against the same hardware. Stopping or removing units must retain
captures and LeRobot calibration files. Support the arm before releasing torque.
See the complete [SO-101 workflow](../../docs/so101-robot-lab.md) before enabling motors.
