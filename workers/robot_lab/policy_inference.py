"""Isolated local ACT inference. This process has no serial or motor interface."""

import base64
import contextlib
import io
import json
import math
import os
import sys
from pathlib import Path

from policy_contract import validate_act_config, validate_processors


def main():
    output = sys.stdout
    checkpoint, device = Path(sys.argv[1]), sys.argv[2]
    validate_processors(checkpoint)
    validate_act_config(checkpoint)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from PIL import Image

        torch.set_num_threads(2)
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        config.device = device
        if hasattr(config, "pretrained_backbone_weights"):
            config.pretrained_backbone_weights = None
        policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True)
        policy.to(device).eval()
        pre, post = make_pre_post_processors(
            policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        camera = next(iter(policy.config.image_features))
        shape = list(policy.config.image_features[camera].shape)
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError("ACT camera must have three RGB channels")

        def infer(message):
            image = Image.open(io.BytesIO(base64.b64decode(message["jpeg"], validate=True)))
            image = image.convert("RGB").resize((shape[2], shape[1]))
            tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
            batch = {
                "observation.state": torch.tensor(message["state"], dtype=torch.float32).unsqueeze(
                    0
                ),
                camera: tensor.unsqueeze(0),
            }
            with torch.inference_mode():
                action = post(policy.select_action(pre(batch))).detach().cpu().reshape(-1).tolist()
            if len(action) != 6 or not all(math.isfinite(x) for x in action):
                raise ValueError("Policy produced a non-finite or incompatible action")
            return action

        # Warm the model using a synthetic frame without touching the hardware.
        buffer = io.BytesIO()
        Image.new("RGB", (shape[2], shape[1])).save(buffer, "JPEG")
        infer({"state": [0.0] * 6, "jpeg": base64.b64encode(buffer.getvalue()).decode()})
        policy.reset()
    output.write(
        json.dumps({"ready": True, "device": device, "camera_key": camera, "shape": shape}) + "\n"
    )
    output.flush()
    for raw in sys.stdin:
        if len(raw) > 4 * 1024**2:
            raise ValueError("Policy observation exceeds input bound")
        message = json.loads(raw)
        with contextlib.redirect_stdout(sys.stderr):
            if message.get("reset") is True:
                policy.reset()
                reply = {"reset": True}
            else:
                reply = {"action": infer(message)}
        output.write(json.dumps(reply, allow_nan=False) + "\n")
        output.flush()


if __name__ == "__main__":
    main()
