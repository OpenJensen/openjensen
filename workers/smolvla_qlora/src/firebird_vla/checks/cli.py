"""Check QLoRA against the six LIBERO checkpoints from docs/idea/18."""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from firebird_vla.checkpoint import sha256, write_json

from .catalog import PROFILES, load_catalog, result_row
from .fixtures import validate_fixture


def runtime_blockers(entry, runtime):
    reasons = []
    if not runtime:
        return ["Native environment and matching training fixture have not been supplied"]
    for key in ("python", "fixture", "dependency_lock"):
        value = runtime.get(key)
        if not value or not Path(value).exists():
            reasons.append(f"Missing {key} path")
    if entry["id"] not in ("smolvla", "pi0"):
        if not runtime.get("native_repo") or not Path(runtime["native_repo"]).is_dir():
            reasons.append("Missing pinned native source checkout")
        if not re.fullmatch(r"[0-9a-f]{40}", runtime.get("native_code_commit") or ""):
            reasons.append("Missing immutable native code commit")
    if entry["id"] == "pi05":
        for key in ("converted_checkpoint", "conversion_manifest"):
            if not runtime.get(key) or not Path(runtime[key]).exists():
                reasons.append(f"Missing {key}; JAX weights cannot be passed to bitsandbytes")
    if entry["id"] == "gr00t_n17" and not re.fullmatch(
        r"[0-9a-f]{40}",
        runtime.get("backbone_revision") or "",
    ):
        reasons.append("Missing immutable Cosmos-Reason2-2B backbone revision")
    if not reasons:
        try:
            manifest = json.loads((Path(runtime["fixture"]) / "fixture.json").read_text())
            validate_fixture(manifest, entry)
        except (ValueError, OSError, KeyError) as error:
            reasons.append(str(error))
    return reasons


def summarize_phases(row, phases):
    row["phases"] = phases
    qlora_ok = all(phases.get(p, {}).get("status") == "passed" for p in ("qlora", "reload"))
    float_ok = phases.get("float", {}).get("status") == "passed"
    row["qlora_verified"] = qlora_ok
    if qlora_ok and float_ok:
        row["status"] = "qlora_diagnostic_passed"
        row["reason"] = (
            "Native loss/backward/update and loss reload passed; task quality unmeasured"
        )
    elif qlora_ok:
        row["status"] = "incomplete_float_reference"
        row["reason"] = "QLoRA check passed, but the floating-point diagnostic did not"
    else:
        statuses = {p["status"] for p in phases.values()}
        row["status"] = next(
            (s for s in ("oom", "timeout", "failed", "blocked") if s in statuses), "incomplete"
        )
        row["reason"] = "; ".join(
            f"{p}: {v.get('reason', v['status'])}"
            for p, v in phases.items()
            if v["status"] != "passed"
        )
    # These remain native diagnostic measurements, not task success or deployment size/latency.
    row["metrics"] = {
        "native_float_loss": phases.get("float", {}).get("native_loss"),
        "nf4_loss_before": phases.get("qlora", {}).get("native_loss_before"),
        "nf4_loss_after": phases.get("qlora", {}).get("native_loss_after"),
        "reload_loss_abs_difference": phases.get("reload", {}).get("reload_abs_difference"),
        "qlora_peak_allocated_bytes": phases.get("qlora", {}).get("peak_allocated_bytes"),
        "task_success": None,
        "attempted_episodes": None,
        "action_chunk_p95_ms": None,
    }
    return row


def run_entry(entry, runtime, output, args):
    output.mkdir()
    job = {
        "entry": entry,
        "runtime": runtime,
        "output": str(output),
        "steps": args.steps,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "dependency_lock_sha256": sha256(runtime["dependency_lock"]),
    }
    job_path = output / "job.json"
    write_json(job_path, job)
    phases = {}
    env = dict(os.environ)
    # The same worker source is used in each isolated environment; no native source monkey-patches.
    source = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    for phase in ("float", "qlora", "reload"):
        if phase == "reload" and phases["qlora"]["status"] != "passed":
            phases[phase] = {"status": "blocked", "reason": "No successful QLoRA adapter to reload"}
            write_json(output / f"{phase}.json", phases[phase])
            continue
        with (output / f"{phase}.log").open("w") as stream:
            try:
                result = subprocess.run(
                    [
                        str(Path(runtime["python"]).absolute()),
                        "-m",
                        "firebird_vla.checks.worker",
                        str(job_path),
                        phase,
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=args.timeout_seconds,
                    check=False,
                )
                path = output / f"{phase}.json"
                evidence = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {
                        "status": "failed",
                        "reason": f"Worker exited {result.returncode}; see {phase}.log",
                    }
                )
                if result.returncode and evidence["status"] == "passed":
                    evidence = {"status": "failed", "reason": "Worker exited unsuccessfully"}
                phases[phase] = evidence
            except subprocess.TimeoutExpired:
                phases[phase] = {
                    "status": "timeout",
                    "reason": "Phase exceeded the declared time budget",
                }
            except OSError as error:
                phases[phase] = {"status": "blocked", "reason": str(error)}
        write_json(output / f"{phase}.json", phases[phase])
    row = summarize_phases(result_row(entry), phases)
    row["evidence_refs"] = [str(job_path), *[str(output / f"{p}.json") for p in phases]]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="configs/benchmarks/open_weight_vlas.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--inspect", action="store_true", help="Read small public checkpoint metadata"
    )
    modes.add_argument("--run", action="store_true", help="Run isolated native CUDA checks")
    parser.add_argument(
        "--runtimes", help="JSON mapping of model IDs to native environments and fixtures"
    )
    parser.add_argument("--models", nargs="+", choices=list(PROFILES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if min(args.steps, args.rank, args.timeout_seconds) < 1 or not 0 < args.learning_rate < 1:
        parser.error("steps, rank, timeout must be positive; learning rate must be in (0,1)")
    catalog = load_catalog(args.catalog)
    runtimes = json.loads(Path(args.runtimes).read_text()) if args.runtimes else {}
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = set(args.models or [e["id"] for e in catalog["models"]])
    entries = sorted(catalog["models"], key=lambda e: e["priority"])
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog_sha256": sha256(args.catalog),
        "scope": "QLoRA integration diagnostics",
        "closed_loop_benchmark_executed": False,
        "rows": [result_row(entry) for entry in entries],
    }
    write_json(output / "report.json", report)
    for index, entry in enumerate(entries):
        row = result_row(entry)
        if entry["id"] not in selected:
            row["reason"] = "Not selected for this run; remains in the six-model scope"
        elif args.inspect:
            from .inspect import inspect_entry

            try:
                row["inspection"] = inspect_entry(entry)
                row["status"] = row["inspection"]["status"]
                row["reason"] = row["inspection"]["reason"]
            except Exception as error:
                row.update(status="inspection_failed", reason=str(error))
        elif args.run:
            runtime = runtimes.get(entry["id"], {})
            blockers = runtime_blockers(entry, runtime)
            if blockers:
                row.update(status="blocked", reason="; ".join(blockers))
            else:
                report["rows"][index].update(
                    status="running", reason="Native diagnostic in progress"
                )
                write_json(output / "report.json", report)
                try:
                    row = run_entry(entry, runtime, output / entry["id"], args)
                except KeyboardInterrupt:
                    report["rows"][index].update(
                        status="interrupted", reason="User interrupted diagnostic"
                    )
                    write_json(output / "report.json", report)
                    raise
        else:
            row["reason"] = (
                "Planned only; supply the native environment and fixture, then use --run"
            )
        report["rows"][index] = row
        write_json(output / "report.json", report)
        print(f"{entry['label']}: {row['status']} — {row['reason']}", flush=True)
    if args.run and any(
        r["model_id"] in selected and r["status"] != "qlora_diagnostic_passed"
        for r in report["rows"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
