"""Isolated, offline native policy inference. This process has no motor interface."""

import base64
import contextlib
import io
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

from policy_contract import validate_policy_config, validate_processors, validate_smolvla_assets


def main():
    output = sys.stdout
    checkpoint, device = Path(sys.argv[1]), sys.argv[2]
    document = validate_policy_config(checkpoint)
    kind = document["type"]
    validate_processors(checkpoint, kind)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if device == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.35")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from PIL import Image

        if kind == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy as Policy
        elif kind == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Policy
        else:
            raise ValueError("This native policy adapter is unsupported")
        torch.set_num_threads(2)
        if device == "mps":
            torch.mps.set_per_process_memory_fraction(0.25)
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        config.device = device
        overrides = {"device_processor": {"device": device}}
        if kind == "act":
            config.pretrained_backbone_weights = None
        else:
            assets = validate_smolvla_assets(checkpoint)
            # Full SmolVLA checkpoints contain the VLM weights. Its constructor
            # needs only a closed local configuration/tokenizer asset bundle.
            config.vlm_model_name = str(assets)
            config.load_vlm_weights = False
            overrides["tokenizer_processor"] = {"tokenizer_name": str(assets)}
        policy = Policy.from_pretrained(
            checkpoint, config=config, local_files_only=True, strict=True
        )
        policy.to(device).eval()
        pre, post = make_pre_post_processors(
            policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides=overrides,
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        cameras = list(policy.config.image_features)
        camera = sys.argv[3] if len(sys.argv) > 3 else cameras[0]
        if camera not in cameras:
            raise ValueError("Selected policy camera is unsupported")
        shape = list(policy.config.image_features[camera].shape)

        def picture(encoded, key):
            image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
            if image.width * image.height > 24_000_000:
                raise ValueError("Choose an observation image smaller than 24 megapixels")
            image = image.convert("RGB")
            image.thumbnail((1280, 1024))
            if kind == "act":
                size = policy.config.image_features[key].shape
                image = image.resize((size[2], size[1]))
            tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
            return tensor.unsqueeze(0)

        def infer(message):
            state = message.get("state")
            if (
                not isinstance(state, list)
                or len(state) != 6
                or any(type(v) not in {int, float} or not math.isfinite(v) for v in state)
            ):
                raise ValueError("Policy observation needs six finite joint values")
            images = message.get("images")
            if images is None and (len(cameras) == 1 or message.get("repeat") is True):
                # A one-camera policy, or an offline preview that knowingly shows one
                # picture to every camera input. An arm run never takes this path with
                # several cameras: it must send each trained view.
                images = {key: message["jpeg"] for key in cameras}
            if not isinstance(images, dict) or set(images) != set(cameras):
                raise ValueError("Policy observation needs one image for every trained camera")
            batch = {"observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0)}
            for key in cameras:
                batch[key] = picture(images[key], key)
            if kind == "smolvla":
                task = message.get("task", "")
                if not isinstance(task, str) or not task.strip() or len(task) > 1000:
                    raise ValueError("SmolVLA needs a task instruction of at most 1000 characters")
                batch["task"] = task
            before = time.monotonic()
            with torch.inference_mode():
                action = post(policy.select_action(pre(batch))).detach().cpu().reshape(-1).tolist()
            if len(action) != 6 or not all(math.isfinite(x) for x in action):
                raise ValueError("Policy produced a non-finite or incompatible action")
            return action, (time.monotonic() - before) * 1000

        buffer = io.BytesIO()
        Image.new("RGB", (shape[2], shape[1])).save(buffer, "JPEG")
        _, warm_ms = infer(
            {
                "state": [0.0] * 6,
                "jpeg": base64.b64encode(buffer.getvalue()).decode(),
                "repeat": True,
                "task": "Remain still.",
            }
        )
        policy.reset()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        info = {
            "ready": True,
            "device": device,
            "policy_type": kind,
            "camera_key": camera,
            "camera_keys": cameras,
            "shape": shape,
            "task_required": kind == "smolvla",
            "warm_inference_ms": round(warm_ms, 1),
            "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            "mps_driver_bytes": torch.mps.driver_allocated_memory() if device == "mps" else None,
        }
    output.write(json.dumps(info) + "\n")
    output.flush()
    for raw in sys.stdin:
        if len(raw) > 4 * 1024**2:
            raise ValueError("Policy observation exceeds input bound")
        try:
            message = json.loads(raw)
            if not isinstance(message, dict):
                raise ValueError("Supply a policy observation object")
            with contextlib.redirect_stdout(sys.stderr):
                if message.get("reset") is True:
                    policy.reset()
                    reply = {"reset": True}
                else:
                    action, elapsed = infer(message)
                    reply = {"action": action, "inference_ms": round(elapsed, 1)}
        except (ValueError, TypeError, KeyError, OSError) as error:
            policy.reset()
            reply = {"error": str(error)[:300]}
        output.write(json.dumps(reply, allow_nan=False) + "\n")
        output.flush()


if __name__ == "__main__":
    main()
