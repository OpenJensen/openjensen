# Dataset preview stills

These JPEG stills are extracted from the first frame of public source videos and resized to 640px width. They are real source camera frames, not generated illustrations.

- `so101-pickup.jpg`: codywang/so101_pickup_test, revision `ecef85bc07005f771ad86deeff1427f9d72953ed`, camera `observation.images.front`. Source: https://huggingface.co/datasets/codywang/so101_pickup_test/tree/ecef85bc07005f771ad86deeff1427f9d72953ed
- `so100-pickplace.jpg`: lerobot/svla_so100_pickplace, revision `728583b5eaf9e739a7f119e2def466fa1d552402`, camera `observation.images.top`. Source: https://huggingface.co/datasets/lerobot/svla_so100_pickplace/tree/728583b5eaf9e739a7f119e2def466fa1d552402

For each source above the exact video path is `videos/<camera>/chunk-000/file-000.mp4` at the listed revision. The stills use time 0; the extraction is `ffmpeg -i <pinned-video-url> -frames:v 1 -vf scale=640:-2 -q:v 3 <poster>.jpg`.

Both source dataset cards declare Apache-2.0 at these revisions. The license text is included in `LICENSE-APACHE-2.0.txt`. The datasets and their contributors retain their original ownership; these stills are included for attribution-backed dataset selection.

## Source metadata

Both pinned revisions declare LeRobot v3.0. Counts come from the source `meta/info.json` at those revisions.

| Dataset | Episodes | Frames | Camera keys (under `observation.`) |
| --- | ---: | ---: | --- |
| SO-101 pickup | 30 | 4,500 | `images.front` |
| SO-100 pick & place | 50 | 19,631 | `images.top`, `images.wrist` |

To refresh a starter, resolve its Hub SHA, inspect and preview that exact SHA through the backend, update the counts and camera keys, regenerate its still, and update this record together with `src/lib/dataset-starters.ts`. Keep an immutable commit in the catalogue; do not silently replace it with `main`.
