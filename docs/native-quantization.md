# Quantize ACT in the web app

1. Install and register the [local ACT packing worker](native-act-quantization.md).
2. Export a complete ACT training checkpoint or import a complete native ACT inference policy.
3. Open **Quantize → ACT** and select that saved policy.
4. Choose INT8 or INT4, review the recipe, and start.
5. Open the saved job to follow events and download the packed package.

Use an input with one camera, six state/action coordinates, 100-action chunks, `use_vae=false`, and saved config/processors/statistics. Set an integer timeout from 30–600 seconds; the default is 600.

To replay the output, choose **Replay recorded observations**, configure the [replay worker](../workers/isaac_sim/NATIVE_REPLAY.md) and select the input observations. For SmolVLA, choose its separate card and follow [GGUF setup](policy-workflow.md).

Cancel the selected job from its history. If a response is lost or the connection fails, refresh the project job list and inspect the saved request before submitting again.
