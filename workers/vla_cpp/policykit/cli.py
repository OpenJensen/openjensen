from __future__ import annotations

import argparse
import json
import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from .benchmark import BenchmarkRunner, artifact_path, file_hash
from .metrics import timed_stage
from .config import load_config
from .conversion import cache_matches, conversion_identity, convert_atomic
from .leaderboard import render
from .runtime import VlaCpp, hardware_fingerprint, run

DEFAULT_CONFIG = "configs/benchmark.v1.yaml"


def prepare(config_path: str) -> None:
    config = load_config(config_path)
    runtime = VlaCpp(config)
    runtime.check_tools()
    runtime.ensure_prepared()
    try:
        from huggingface_hub import snapshot_download, model_info
    except ImportError as exc:
        raise RuntimeError("Install preparation dependencies: pip install -e '.[prepare]'") from exc
    lock = {"runtime_tag": config.data["runtime"]["tag"], "hardware": hardware_fingerprint(), "models": {}}
    for name, model in config.data["models"].items():
        print(f"[prepare] Resolving and downloading {name} from {model['source']} …", flush=True)
        info = model_info(model["source"], revision=model["revision"])
        local_dir = config.artifacts / "sources" / name
        snapshot_download(repo_id=model["source"], revision=info.sha, local_dir=local_dir)
        lock["models"][name] = {"source": model["source"], "requested_revision": model["revision"], "resolved_revision": info.sha, "path": str(local_dir)}
    print(runtime.write_lock(lock))


def quantize(config_path: str, model_name: str, preset: str) -> None:
    config = load_config(config_path)
    runtime = VlaCpp(config)
    runtime.require_prepared()
    config.model(model_name)
    config.preset(preset)
    source_dir = config.artifacts / "sources" / model_name
    bf16 = artifact_path(config, model_name, "bf16")
    bf16.parent.mkdir(parents=True, exist_ok=True)
    destination = artifact_path(config, model_name, preset)
    timing_path = destination.with_suffix('.timing.json')
    timings = {}
    record = {"schema_version": 1, "model": model_name, "preset": preset,
              "hardware": hardware_fingerprint(), "status": "running",
              "conversion_reused": False, "timings": timings}
    try:
        with timed_stage(timings, "module_wall_seconds"):
            with timed_stage(timings, "conversion_cache_check_seconds"):
                identity = conversion_identity(config, runtime, model_name, source_dir, bf16)
                record["conversion_identity"] = identity
                record["conversion_reused"] = cache_matches(bf16, bf16.with_suffix('.conversion.json'), identity)
            if record["conversion_reused"]:
                timings["conversion_seconds"] = 0.0
            else:
                with timed_stage(timings, "conversion_seconds"):
                    convert_atomic(config, runtime, model_name, source_dir, bf16, identity, run)
            if preset != "bf16":
                with timed_stage(timings, "quantize_seconds"):
                    run(runtime.quantize_command(bf16, destination, config.preset(preset)), cwd=runtime.source_dir)
            else:
                timings["quantize_seconds"] = 0.0
            with timed_stage(timings, "artifact_hash_seconds"):
                record["sha256"] = file_hash(destination)
        record["status"] = "complete"
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        timing_path.write_text(json.dumps(record, indent=2) + "\n")
    print(destination)


def main() -> None:
    parser = argparse.ArgumentParser(prog="policykit", description="Local VLA quantization benchmarks")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    q = sub.add_parser("quantize")
    q.add_argument("--model", required=True, choices=("smolvla", "pi0"))
    q.add_argument("--preset", required=True, choices=("bf16", "q8_0", "q4_0", "q8_0_vision", "q4_0_vision"))
    b = sub.add_parser("benchmark")
    b.add_argument("--run", required=True)
    lb = sub.add_parser("leaderboard")
    lbsub = lb.add_subparsers(dest="leaderboard_command", required=True)
    build = lbsub.add_parser("build")
    build.add_argument("--run", required=True)
    serve = lbsub.add_parser("serve")
    serve.add_argument("--run", required=True)
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.config)
        elif args.command == "quantize":
            quantize(args.config, args.model, args.preset)
        elif args.command == "benchmark":
            print(BenchmarkRunner(load_config(args.config), args.run).execute())
        else:
            config = load_config(args.config)
            root = config.artifacts / "leaderboards" / args.run
            page = render(config.artifacts / "runs" / args.run / "results.json", root / "index.html")
            if args.leaderboard_command == "build":
                print(page)
            else:
                os.chdir(root)
                print(f"Serving {page} at http://127.0.0.1:{args.port}")
                ThreadingHTTPServer(("127.0.0.1", args.port), SimpleHTTPRequestHandler).serve_forever()
    except RuntimeError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
