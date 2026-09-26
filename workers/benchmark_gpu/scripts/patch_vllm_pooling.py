"""Forward prompt embeddings in the isolated vLLM 0.9.2 V0 pooling runner."""

import argparse
from importlib.metadata import distribution
from pathlib import Path

OLD = (
    "            input_ids=model_input.input_tokens,\n"
    "            positions=model_input.input_positions,"
)
NEW = (
    "            input_ids=model_input.input_tokens,\n"
    "            inputs_embeds=model_input.inputs_embeds,\n"
    "            positions=model_input.input_positions,"
)


def patch(path):
    original = path.read_text()
    if NEW in original:
        return False
    if original.count(OLD) != 1:
        raise RuntimeError("Unexpected pooling runner source; refusing to patch")
    backup = path.with_suffix(path.suffix + ".original")
    if backup.exists() and backup.read_text() != original:
        raise RuntimeError("Existing backup differs from source")
    backup.write_text(original)
    path.write_text(original.replace(OLD, NEW, 1))
    return True


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    dist = distribution("vllm")
    if dist.version != "0.9.2":
        raise RuntimeError("This source patch is only validated for vLLM 0.9.2")
    path = Path(dist.locate_file("vllm/worker/pooling_model_runner.py"))
    print("patched" if patch(path) else "already patched", path)


if __name__ == "__main__":
    main()
