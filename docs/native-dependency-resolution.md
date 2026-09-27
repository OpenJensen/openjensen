# Native GPU dependency resolution

All 13 native LeRobot profiles resolved successfully for **Linux x86_64 / Python
3.12**, using the deployment's **uv 0.12.19** and exact CUDA 12.8 constraints.
No dependency conflicts were found, so no worker profile or setup changes were
needed. The pinned source is
`e595b7902714ba51f91e47523f66f89c5181b649`.

The resolver preserved `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128` and
`torchcodec==0.11.1+cu128` in every profile. Common final-environment requirements
`google-cloud-storage==3.4.1` and `setuptools==80.10.2` were included in each
resolution. This checked the combined final dependency graph, including the
policy-specific extra.

| Policy | LeRobot extras | Packages | Result |
| --- | --- | ---: | --- |
| ACT | `training` | 109 | Resolved |
| Diffusion Policy | `training,diffusion` | 113 | Resolved |
| EO-1 | `training,eo1` | 120 | Resolved |
| EVO-1 | `training,evo1` | 119 | Resolved |
| GR00T N1.7 | `training,groot` | 128 | Resolved |
| Multi-Task DiT | `training,multi_task_dit` | 122 | Resolved |
| π₀ | `training,pi` | 120 | Resolved |
| π₀-FAST | `training,pi` | 120 | Resolved |
| π₀.₅ | `training,pi` | 120 | Resolved |
| VLA-JEPA | `training,vla_jepa` | 123 | Resolved |
| VQ-BeT | `training` | 109 | Resolved |
| WALL-X | `training,wallx` | 123 | Resolved |
| XVLA | `training,xvla` | 119 | Resolved |

These are metadata-only `uv pip compile` runs, in batches of up to three. Builds
and Python downloads were disabled; no packages were installed and no GPUs or
model downloads were launched. The cache audit found no new wheel archives or
cached files larger than 8 MiB. This establishes dependency compatibility, not
model execution or training quality. ACT's separate live GPU verification is in
[the cloud training verification record](cloud-training-verification.md).

The [machine-readable matrix](native-dependency-resolution.json) records exact
critical versions, source hash and per-profile resolution hashes. Full compiled
requirements are preserved in [`evidence/native-dependencies/`](evidence/native-dependencies/).

To reproduce, put the selected profile's pinned `lerobot[extras] @ git+...`
requirement and the two common requirements in `profile.in`, with the two CUDA
pins in `native-cu128-constraints.txt`, then run:

```sh
uv pip compile --no-build --no-python-downloads \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --constraint native-cu128-constraints.txt \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match profile.in
```

Use uv 0.12.19, matching the generated cloud setup. Resolution timestamps and
hashes describe this audit; future unrestricted transitive releases can change
the result even while the upstream source and CUDA versions remain pinned.
