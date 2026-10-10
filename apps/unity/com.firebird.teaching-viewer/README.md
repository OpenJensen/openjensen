# OPEN JENSEN teaching viewer

A receive-only Unity package for the teaching worker's atomic `GET /frame` version 1 response. It shows the RGB camera image and explicitly named joint positions in radians. It does not write transforms, articulation targets, robot commands, LiveKit rooms, or cloud settings. Adding the component never starts a connection.

This package is **not** a replacement for the supplied Unity project, a LiveKit data receiver, a calibrated 3D robot animation, or proof of simulator/robot task success.

## Install and bind

Use Unity **6000.6** and install this directory through Package Manager → Add package from disk → `package.json`. Declared dependencies: Unity Newtonsoft JSON 3.2.2 plus built-in image conversion and IMGUI 1.0.0. Transport uses the .NET HTTP client.

Create an otherwise empty GameObject with `Firebird.TeachingViewer.TeachingViewer`. From a trusted runtime integration, call:

```csharp
var binding = new ViewBinding(
    expectedSessionId, "observation.images.front", expectedCameraPrim,
    expectedWidth, expectedHeight, trustedOrderedJointLimits,
    generatedFixture: false, staleAfterSeconds: 2);
viewer.Configure("http://127.0.0.1:PORT", runtimeBearerToken, binding);
viewer.StartViewing(); // explicit operator action; never automatic
// Later: viewer.StopViewing();
```

All values must come from independently selected session/controller configuration, not from an incoming packet. Joint limits are measured-articulation radian limits, including the gripper's articulation coordinate; these are **not** LeRobot's normalized gripper coordinates. Do not guess signs, zeros, order, units, or camera orientation from the prototype's joint names.

The bearer token stays in a private, nonserialized runtime field and is cleared when the component is disabled. Supply it through trusted runtime code; do not put it in a scene, Inspector field, example source, or log. The token may authorize more than viewing at the worker, so this client intentionally exposes only `GET /frame`. It is not a separate least-privilege credential authority.

Only the literal `http://127.0.0.1:PORT` origin is accepted. Redirects and environment proxies are disabled. A remote worker requires an **already established operator-controlled loopback tunnel**; this package does not create one. It does not connect to the existing prototype's LiveKit room or interpret its binary messages. Keep the old actuation receivers out of a viewer-only scene; this package neither disables nor makes other components safe.

## Observation and freshness rules

The decoder requires the complete, exact available-frame schema, matching expected session, camera, dimensions, ordered joint names and `units: rad`. It validates finite state against supplied bounds, canonical RGB base64 and SHA-256, integer identity fields, and equality of frozen acquisition context with `current_context`. Missing/unavailable, malformed, oversized, reordered, superseded, or out-of-range frames are rejected. Duplicate fields, extra fields, non-UTF-8 bytes and trailing JSON are rejected.

The body bound derives from the configured image dimensions (at most 1920×1920) plus 64 KiB of metadata. Each HTTP request has a three-second cancellation deadline. Only one request runs at a time, followed by a 0.1-second pause; this is not a guaranteed frame rate. Stop/disable cancels the outstanding request.

A repeated capture never renews freshness. A rejected response retires the last capture until a newer valid observation arrives; replaying a formerly valid packet cannot undo a pause/context rejection. Transport disconnection can recover using an otherwise valid duplicate only while its original age remains within the freshness bound. New but already stale captures never relabel the last displayed image's metadata.

Displayed conservative age is the worker's `source_age_ns` plus the full local HTTP round trip and elapsed local time. Absolute monotonic clocks from different hosts are never compared. Source age starts when the executor receives the observation; it is **not camera exposure age**. The worker may deliberately retain a prior image after pause/task changes, making it stale until another real observation exists.

The UI keeps the last accepted image visible with an explicit **STALE / DISCONNECTED** label. Source age, HTTP round trip, and simulation time are distinct. Model latency says **not measured**. Set `generatedFixture: true` for generated input; the banner clearly states it is not simulator or robot acceptance.

## Verification

Editor tests require Unity Test Framework 1.8.0 and this package in the temporary project's `testables` list. The project also needs built-in `com.unity.modules.imageconversion` and `com.unity.modules.imgui` version 1.0.0. Example `Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.firebird.teaching-viewer": "file:/absolute/path/to/com.firebird.teaching-viewer",
    "com.unity.nuget.newtonsoft-json": "3.2.2",
    "com.unity.test-framework": "1.8.0",
    "com.unity.modules.imageconversion": "1.0.0",
    "com.unity.modules.imgui": "1.0.0"
  },
  "testables": ["com.firebird.teaching-viewer"]
}
```

Run EditMode and PlayMode separately using the actual installed editor executable:

```sh
"$UNITY_EDITOR" -batchmode -nographics -projectPath "$DISPOSABLE_PROJECT" \
  -runTests -testPlatform EditMode -testResults "$OUTPUT/edit.xml" -logFile "$OUTPUT/edit.log"
"$UNITY_EDITOR" -batchmode -nographics -projectPath "$DISPOSABLE_PROJECT" \
  -runTests -testPlatform PlayMode -testResults "$OUTPUT/play.xml" -logFile "$OUTPUT/play.log"
```

The cross-component EditMode test additionally consumes an explicitly generated teaching-worker `Session.prepare` → mailbox frame. Set `FIREBIRD_UNITY_FRAME_FIXTURE` to that JSON and `FIREBIRD_UNITY_EXPECTED_SESSION` to its independently recorded session. Its fixed synthetic setup is 32×32 RGB, `/World/Camera`, six `joint_0`…`joint_5` coordinates at zero with [-1,1] limits. Without the fixture path this one test is explicitly skipped; do not report that run as cross-component verification.

Tests cover contract rejection, context invalidation/replay, freshness and metadata pairing, a real local HTTP GET with an artificial token, generated RGB texture bytes, no auto-connect, no transform changes, and disconnect status. They use no Isaac process, hardware, real credential, provider API, or LiveKit room.

Live Isaac rendering orientation, LiveKit transport/publisher identity, calibrated SO-101 pose visualization, deployed-player/AOT builds, Windows/Linux Unity editors, interactive visual inspection, and latency/frame-rate performance under real streams require separate validation before release.
