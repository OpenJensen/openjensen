# Quantize ACT in the web app

Pack a saved ACT policy to INT8 or INT4 from **Quantize**, then download, replay or simulate the package.

## Before you start

- Install and register the [local ACT packing worker](native-act-quantization.md).
- Have a complete ACT inference policy in the project: export a complete ACT training checkpoint, or import a complete native ACT inference policy.
- The policy needs one camera, six state/action coordinates, saved horizons with `1 <= execution <= prediction <= 1024`, `use_vae=false`, and saved config, processors and statistics. See the [full requirements](native-act-quantization.md#prepare-the-source-policy).
- Keep browser storage enabled. The app uses it to recover a request after a reload.

## Quantize

1. Open **Quantize** and select **Start a new quantization**.
2. Under **Choose a model to quantize**, open the ACT model. A training checkpoint shows **Export this ACT checkpoint as an inference model first.**
3. Choose the **Policy**, and the **Compute** worker if there is more than one.
4. Under **Compression**, choose **8-bit** or **4-bit**.
5. Set **Timeout (seconds)** to a whole number from 30 to 600. The default is 600.
6. Select **Create ACT quantized package**.
7. Open the saved job to follow its activity, then select **Download INT8 package** (or **Download INT4 package**).

A SmolVLA model opens the GGUF flow instead; see [GGUF setup](policy-workflow.md#gguf-quantization-and-diagnostics).

## Use the package

- **Replay recorded observations:** configure the [replay worker](../workers/isaac_sim/NATIVE_REPLAY.md) and select the input observations.
- **Prepare simulation:** available when a packed ACT simulation profile is configured. See [Isaac simulation](native-simulation.md).

The job reports stored bytes and action drift on generated observations only. Task quality and speed are not measured.

## Cancel or recover

- To stop a job, select **Cancel selected ACT quantization**, then **Confirm cancellation**.
- If a response is lost or the connection fails, the app pauses new submissions and **Quantize** shows **Review request**. Refresh the project job list and inspect the saved request, then select **I checked the jobs; allow a new request** before submitting again.
