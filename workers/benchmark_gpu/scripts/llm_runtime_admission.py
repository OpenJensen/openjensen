"""Exercise the actual LLM runtime against the pinned full SmolVLA checkpoint.

A load failure is recorded, never relabeled as a policy benchmark. SmolVLM-only
text generation cannot satisfy this check's required continuous action output.
"""

import argparse
import importlib.metadata
import json
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", choices=["vllm", "tensorrt_llm"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = {
        "runtime": args.runtime,
        "full_action_execution": False,
        "stage": "import",
        "required_output": (
            "complete unnormalized 50 x 7 action chunk from observations, state and task"
        ),
    }
    try:
        from huggingface_hub import snapshot_download

        result["version"] = importlib.metadata.version(args.runtime)
        path = snapshot_download(
            "lerobot/smolvla_libero",
            revision="31d453f7edd78c839a8bbc39744a292686daf0de",
            local_files_only=True,
        )
        if args.runtime == "vllm":
            from vllm import LLM
            from vllm.model_executor.models import ModelRegistry

            result["smol_registered_architectures"] = [
                a for a in ModelRegistry.get_supported_archs() if "Smol" in a
            ]
            result["stage"] = "checkpoint_admission"
            model = LLM(
                model=path,
                dtype="half",
                max_model_len=512,
                gpu_memory_utilization=0.15,
                enforce_eager=True,
            )
        else:
            from tensorrt_llm import LLM

            result["stage"] = "checkpoint_admission"
            model = LLM(
                model=path, backend="pytorch", dtype="float16", max_batch_size=1, max_seq_len=512
            )
        result["status"] = "loaded_but_complete_action_execution_unverified"
        del model
    except Exception as exc:
        result.update(
            status="failed",
            exception=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["stage"], result["status"], result.get("error", "")[:600])


if __name__ == "__main__":
    main()
