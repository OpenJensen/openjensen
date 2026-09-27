# Product screenshot provenance

Captured on September 27, 2026 at 11:39 Asia/Yerevan from OPEN JENSEN
(Joint Embodied Neural Simulation & Execution Network) at
`http://127.0.0.1:8000/`, using Playwright 1.63 and Chromium. Each of these three historical PNGs is a
1600 × 1000 desktop viewport at 1× device scale, in the product's light theme.
Fonts and the relevant records or camera frame were loaded before capture.

These captures predate the current navigation and minimal page layout. They remain historical evidence of the captured screens; see the [workspace guide](../workspace-guide.md) for current controls.

Repository HEAD at capture was `58d77e4`, with the OPEN JENSEN branding changes
present in the working tree. Screenshots use the freshly rebuilt frontend
export, build ID `M5R1NVHXku2DeTClMKmIW`. The served page title was
`OPEN JENSEN · Dataset workspace`, and the visible sidebar brand and `OJ`
workspace badge were verified in all three images. The existing backend was
not restarted.

The served index SHA-256 was
`775b411dc7b65f5840785cb01aaeb75613914f812eb494846de67b34e1c79008`.

| File | Actual screen and evidence scope |
| --- | --- |
| `dataset-explorer.png` | A saved inspection of the public [`codywang/so101_pickup_test`](https://huggingface.co/datasets/codywang/so101_pickup_test) dataset: 30 episodes, 4,500 frames, 30 FPS. Episode 0 is paused at two seconds. The page is scrolled to show the camera and shared playback controls. |
| `training-workspace.png` | A genuine completed ACT run recorded by the application: 100 optimizer steps, recorded training and validation loss, and a saved checkpoint. This is training telemetry, not evidence of task success or a VLA benchmark. |
| `native-simulation.png` | The Native Isaac setup form at capture time with a registered SO-101 cup profile and an existing ACT inference export selected. This shows configuration only; no simulation was launched and no successful rollout is implied. |

These are unmodified product screenshots, with no mocked API responses,
fabricated metrics, substituted text, or injected styles. Browser interactions
were limited to reading saved records, navigation, scrolling, camera playback
position, and selecting an existing policy in the form. Non-read HTTP requests
were blocked in the capture browser. No jobs, imports, settings changes, cloud
operations, or uploads were submitted.


## Current model-choice preview

`workspace-models.png` shows the current Distill overview at 1440 × 1000 in the
light theme. It was captured by the desktop model-choice browser test on
September 27, 2026 from this UI revision. The test uses generated project and
worker API records; the image demonstrates layout and explicit selection only.
No training, distillation or cloud job was started for the capture.
