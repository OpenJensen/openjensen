# Run the OPEN JENSEN teaching viewer

## Install and bind

Use Unity **6000.6**. Choose **Package Manager → Add package from disk** and select this directory's `package.json`. Install Unity Newtonsoft JSON 3.2.2, image conversion 1.0.0 and IMGUI 1.0.0.

Start the [teaching worker](../../../workers/teaching/README.md). For a remote worker, establish a loopback tunnel first. Create an empty GameObject and add `Firebird.TeachingViewer.TeachingViewer`.

Supply the session ID, camera prim, image dimensions and ordered joint limits from the session/controller configuration. Use articulation coordinates in radians. Supply the worker bearer token through runtime code, then configure and start viewing:

```csharp
var binding = new ViewBinding(
    expectedSessionId, "observation.images.front", expectedCameraPrim,
    expectedWidth, expectedHeight, trustedOrderedJointLimits,
    generatedFixture: false, staleAfterSeconds: 2);
viewer.Configure("http://127.0.0.1:PORT", runtimeBearerToken, binding);
viewer.StartViewing();
// Later: viewer.StopViewing();
```

Replace `PORT` with the worker's loopback port. Set `generatedFixture: true` when using generated input. If the display shows **STALE / DISCONNECTED**, check the worker connection and current session configuration, then reconnect.

## Verification

Install Unity Test Framework 1.8.0 and add this package to the test project's `testables`. Example `Packages/manifest.json`:

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

Set `UNITY_EDITOR` to the installed editor, `DISPOSABLE_PROJECT` to the test project and `OUTPUT` to a new output directory. Run EditMode and PlayMode separately:

```sh
"$UNITY_EDITOR" -batchmode -nographics -projectPath "$DISPOSABLE_PROJECT" \
  -runTests -testPlatform EditMode -testResults "$OUTPUT/edit.xml" -logFile "$OUTPUT/edit.log"
"$UNITY_EDITOR" -batchmode -nographics -projectPath "$DISPOSABLE_PROJECT" \
  -runTests -testPlatform PlayMode -testResults "$OUTPUT/play.xml" -logFile "$OUTPUT/play.log"
```

For the cross-component test, set `FIREBIRD_UNITY_FRAME_FIXTURE` to a generated teaching-worker mailbox-frame JSON and `FIREBIRD_UNITY_EXPECTED_SESSION` to its session ID. Use 32×32 RGB, `/World/Camera`, and six `joint_0` through `joint_5` coordinates at zero with [-1,1] joint bounds.
