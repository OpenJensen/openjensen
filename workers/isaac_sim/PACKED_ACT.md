# Serve a packed ACT policy

## Complete package and identity

Use the native quantizer's complete `native-quantized` envelope or inner `policy`
directory. Include `config.json`, `encoding.json`, `model.fbq`, both saved
processors, all referenced statistics files and any saved temporal/control
contracts. Keep the package inventory unchanged.

Prepare an isolated ACT CPU runtime with LeRobot **0.6.1**, Torch **2.11.0**,
torchvision **0.26.0** and safetensors **0.8.0**. Local `+cpu` version suffixes
are accepted. Use a ResNet18 ACT recipe with one RGB camera, six state/action
coordinates, VAE/PEFT/AMP/temporal ensembling disabled, and
`1 <= execution <= prediction <= 1024`.

## Local invocation

From the repository root, set `ACT_PYTHON` and `POLICY_DIRECTORY` to your runtime
interpreter and complete policy directory, then inspect:

```sh
export PYTHONPATH="$PWD/workers/isaac_sim:$PWD/workers/firebird_quant/src:$PWD/workers/act_optimizer/src"
# ACT_PYTHON and POLICY_DIRECTORY are operator-owned local paths.
"$ACT_PYTHON" -m sim_worker.rollout.checkpoint_package \
  --checkpoint "$POLICY_DIRECTORY" --inspect-only
```

Set `VERIFIED_MODEL_ID` to the model ID printed by inspection of those exact
bytes, then serve:

```sh
"$ACT_PYTHON" -m sim_worker.rollout.server \
  --host 127.0.0.1 --port 8080 --backend lerobot --device cpu \
  --checkpoint "$POLICY_DIRECTORY" --model-id "$VERIFIED_MODEL_ID" \
  --state-dim 6 --action-steps 100 --camera-key observation.images.front
```

For a downloaded outer envelope, use
[the import command](../skypilot/ROLLOUT.md#import-a-complete-policy-folder-or-downloaded-package)
to obtain the inner policy directory first.

Send `GET /health`, `POST /reset` and `POST /predict` using
[the remote policy protocol](ROLLOUT.md#policy-server). Match camera dimensions
and the saved coordinate order. For an 8/3 temporal recipe, use `--action-steps 8`
to request the full prediction or `--action-steps 3` for its execution prefix.
Stop the server with `Ctrl+C` after the local session.
