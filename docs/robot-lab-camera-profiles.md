# Linux camera capture profiles

The Robot Lab companion prefers MJPEG by default. Some UVC cameras reserve their maximum USB bandwidth even at smaller MJPEG resolutions. A verified alternative transport can be configured by USB vendor/product identity so it follows the camera when Linux renumbers `/dev/video*` nodes.

Set `OPENJENSEN_CAMERA_PROFILES` in the companion service environment to a JSON object. For example, replacing the sample identity with the camera's vendor/product IDs:

```json
{"1234:abcd":{"fourcc":"YUYV","width":800,"height":600,"fps":10}}
```

Each entry requires a four-character uppercase FourCC, width (160–1920), height (120–1080), and integer capture rate (1–60). All cameras with the matching USB identity use that entry; other cameras keep their default transport. Restart the companion after changing the environment.

The companion checks that the driver accepts the selected format, dimensions and rate before streaming. A rejected profile produces an error and releases its camera handle. Preview selection and background recording preserve the configured capture rate. Inventory displays the configured dimensions; recording frame-rate auto-selection still measures actual camera, robot and storage throughput.

A capture profile does not change kernel driver options. If the verified mode needs a UVC bandwidth quirk, configure that separately after testing concurrent capture and image integrity. Keep the previous service environment and driver configuration for rollback.
