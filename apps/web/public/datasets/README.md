# Refresh dataset preview stills

## Sources and attribution

- `so101-pickup.jpg`: [codywang/so101_pickup_test](https://huggingface.co/datasets/codywang/so101_pickup_test/tree/ecef85bc07005f771ad86deeff1427f9d72953ed), revision `ecef85bc07005f771ad86deeff1427f9d72953ed`, camera `observation.images.front`.
- `so100-pickplace.jpg`: [lerobot/svla_so100_pickplace](https://huggingface.co/datasets/lerobot/svla_so100_pickplace/tree/728583b5eaf9e739a7f119e2def466fa1d552402), revision `728583b5eaf9e739a7f119e2def466fa1d552402`, camera `observation.images.top`.

Both dataset cards declare Apache-2.0 at these revisions. Keep `LICENSE-APACHE-2.0.txt` and source attribution with reused imagery. Original ownership remains with the datasets and their contributors.

## Extract a still

Use `videos/<camera>/chunk-000/file-000.mp4` at the pinned revision. Set `PINNED_VIDEO_URL` to that video's URL and `POSTER_PATH` to the new JPEG path, then extract frame zero at 640px width:

```sh
ffmpeg -i "$PINNED_VIDEO_URL" -frames:v 1 -vf scale=640:-2 -q:v 3 "$POSTER_PATH"
```

## Refresh a starter

1. Resolve the dataset's immutable Hub revision.
2. Inspect and preview that revision through the backend.
3. Read episode/frame counts and camera keys from `meta/info.json`.
4. Generate the still and update `src/lib/dataset-starters.ts` with the revision, metadata and image path.
5. Update the source attribution above and keep the immutable revision in the catalogue.
