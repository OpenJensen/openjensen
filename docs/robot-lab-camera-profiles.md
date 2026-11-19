# Linux camera capture profiles

The Robot Lab companion prefers MJPEG by default. Some UVC cameras reserve their maximum USB bandwidth even at smaller MJPEG resolutions. A verified alternative transport can be configured by USB vendor/product identity so it follows the camera when Linux renumbers `/dev/video*` nodes.

Set `OPENJENSEN_CAMERA_PROFILES` in the companion service environment to a JSON object. For example, replacing the sample identity with the camera's vendor/product IDs:

```json
{"1234:abcd":{"fourcc":"YUYV","width":800,"height":600,"fps":10}}
```

Each entry requires a four-character uppercase FourCC, width (160–1920), height (120–1080), and integer capture rate (1–60). All cameras with the matching USB identity use that entry; other cameras keep their default transport. Restart the companion after changing the environment.

The companion checks that the driver accepts the selected format, dimensions and rate before streaming. A rejected profile produces an error and releases its camera handle. Preview selection and background recording preserve the configured capture rate. Inventory displays the configured dimensions; recording frame-rate auto-selection still measures actual camera, robot and storage throughput.

## Two 0abd:8050 cameras on USB 2

Some cameras with USB ID `0abd:8050` request the largest USB reservation for every MJPEG resolution. Linux can then refuse the second stream with `ENOSPC` at stream start even though either camera works alone. The [Linux companion installer](../deploy/install_robot_lab.py) detects two of these cameras on the same high-speed bus. It registers the UVC driver with DKMS using the terminal or desktop administrator prompt. Headless installations require root or noninteractive sudo. Pass `--camera-driver install` to enroll explicitly or `--camera-driver skip` to keep the distribution driver. On Ubuntu the installer can install missing DKMS, build tools and headers from the configured package repositories. Other distributions need these prerequisites and matching sources installed first.

On an existing managed Linux USB host, **Update & restart** checks for these cameras after the new release passes its health check. If enrollment is needed, approve the local administrator prompt. The result appears in the updates dialog. If a camera is in use, the installed module activates on the next reboot. On a headless host without noninteractive sudo, or for a manually managed companion, run `sudo deploy/linux/uvc/install.sh` from the checkout. This enrolls the running kernel and, on systems with a `/boot/vmlinuz` link, the next boot kernel. You can also pass installed kernel release names explicitly. The driver limits the USB reservation to 1536 bytes per microframe only for high-speed MJPEG on the affected model. Existing capture settings stay in effect. DKMS rebuilds the module when new headers are installed. On Secure Boot systems, enroll the distribution's DKMS signing key through its normal enrollment process.

The build prefers matching kernel source under the target module tree. On Ubuntu, if the source is missing, the build uses Git and HTTPS to retrieve the UVC files at the target kernel package's exact Ubuntu tag. An unchanged stock UVC source signature can reuse the local cache. For other distributions or offline builds, install the matching kernel source so that `/lib/modules/<release>/source/drivers/media/usb/uvc/uvc_video.c` exists, or place that kernel's UVC directory under `/usr/src/openjensen-uvc/<release>/drivers/media/usb/uvc/`. A changed upstream driver layout can require an updated patch.

Activate at the next reboot, or finish recording and teleoperation, close camera users, then reload the module with `sudo modprobe -r uvcvideo` and `sudo modprobe uvcvideo`. Verify `/sys/module/uvcvideo/parameters/mjpeg_bandwidth_cap` reads `1536`, then open both previews with their intended MJPEG modes and check each camera's measured frame rate. A loaded module affects all UVC device enumeration during a reload.

To uninstall, run `sudo dkms remove openjensen-uvc/1.0 --all` and reboot to load the distribution driver.
