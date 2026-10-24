# Mac Robot Lab component

For Linux, use the [Linux companion installer](../linux/README.md).

Use the managed installer to install only the camera, servo and LeRobot-format dataset tools on the computer attached to the arm:

```sh
python3 deploy/macos/install_robot_lab.py --source workers/robot_lab --ssh-host gpu-server
```

It installs a private user service and a private SSH connection to the OpenJensen application on your server. It does not install Torch, CUDA, simulation or model weights. An existing profile environment can be reused with `--runtime workers/robot_lab/.venv`, avoiding a second environment.

To inspect an existing installation:

```sh
jensen install robot --check --environment "$HOME/.local/share/jensen/robot-lab/runtime"
```

Use `--offline` when Python and the hardware wheels are cached. A compatible existing PyArrow installation can be reused with `--arrow-site-packages /absolute/path/to/site-packages`; preserve that environment while it is referenced. Tokens and captures stay private. Read the complete [SO-101 workflow and operations guide](../../docs/so101-robot-lab.md) before enabling motors.
