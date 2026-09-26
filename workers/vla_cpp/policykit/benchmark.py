from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import median
from typing import Any

from .config import BenchmarkConfig
from .runtime import VlaCpp, hardware_fingerprint, run
from .selection import choose_candidates
from .metrics import comparisons, timed_stage


@dataclass
class Result:
    model: str
    preset: str
    status: str = "planned"
    artifact: str | None = None
    artifact_bytes: int | None = None
    sha256: str | None = None
    p50_ms: float | None = None
    p90_ms: float | None = None
    peak_rss_mb: float | None = None
    quantize_seconds: float | None = None
    conversion_seconds: float | None = None
    module_wall_seconds: float | None = None
    load_ms: float | None = None
    engine_process_seconds: list[float] | None = None
    cold_load_ok: bool | None = None
    smoke_episodes: int = 0
    rollout_episodes: int = 0
    successes: int = 0
    error: str | None = None
    evidence: list[str] | None = None

    @property
    def success_rate(self) -> float | None:
        return self.successes / self.rollout_episodes if self.rollout_episodes else None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["success_rate"] = self.success_rate
        return data


def artifact_path(config: BenchmarkConfig, model: str, preset: str) -> Path:
    return config.artifacts / "models" / model / f"{model}-{preset}.gguf"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bench(markdown: str) -> tuple[float | None, float | None]:
    rows = [line for line in markdown.splitlines() if line.strip().startswith("|")]
    if len(rows) == 1:
        cells = [cell.strip() for cell in rows[0].strip().strip("|").split("|")]
        if len(cells) == 8:
            try:
                return float(cells[5]), float(cells[6])
            except ValueError:
                return None, None
    if len(rows) < 3:
        return None, None
    header = [cell.strip().lower() for cell in rows[0].strip("|").split("|")]
    values = [cell.strip() for cell in rows[-1].strip("|").split("|")]
    lookup = dict(zip(header, values))
    def numeric(key: str) -> float | None:
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", lookup.get(key, ""))
        return float(match.group()) if match else None
    return numeric("p50 ms"), numeric("p90 ms")


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, run_id: str):
        self.config = config
        self.run_id = run_id
        self.runtime = VlaCpp(config)
        self.output_dir = config.artifacts / "runs" / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def result_path(self) -> Path:
        return self.output_dir / "results.json"

    def load_results(self) -> list[Result]:
        if not self.result_path().exists():
            return [Result(model=model, preset=preset) for model, preset in self.config.matrix()]
        records = json.loads(self.result_path().read_text())["results"]
        return [Result(**{key: value for key, value in record.items() if key != "success_rate"}) for record in records]

    def save(self, results: list[Result]) -> Path:
        payload = {
            "run_id": self.run_id,
            "machine_label": self.config.data["machine_label"],
            "hardware": hardware_fingerprint(),
            "runtime_tag": self.config.data["runtime"]["tag"],
            "results": [result.as_dict() for result in results],
        }
        payload["baseline_comparisons"] = comparisons(payload["results"])
        payload["deployment_selection"] = choose_candidates(
            payload["results"], self.config.data.get("selection", {}).get("max_success_drop_pp", 5.0)
        )
        self.result_path().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return self.result_path()

    def collect_engine_metrics(self, result: Result) -> None:
        path = artifact_path(self.config, result.model, result.preset)
        result.artifact = str(path)
        if not path.exists():
            result.status, result.error = "failed", f"Missing artifact: {path}"
            return
        result.artifact_bytes, result.sha256 = path.stat().st_size, file_hash(path)
        # Recollection must not retain measurements for a replaced artifact.
        result.quantize_seconds = result.conversion_seconds = result.module_wall_seconds = None
        result.p50_ms = result.p90_ms = result.load_ms = None
        result.error = None
        try:
            timing_path = path.with_suffix('.timing.json')
            if timing_path.exists():
                timing = json.loads(timing_path.read_text())
                if timing.get('status') == 'complete' and timing.get('sha256') == result.sha256:
                    for key in ('quantize_seconds', 'conversion_seconds', 'module_wall_seconds'):
                        setattr(result, key, timing['timings'].get(key))
            samples: list[tuple[float | None, float | None]] = []
            loads = []
            result.engine_process_seconds = []
            for repetition in range(3):
                elapsed = {}
                try:
                    with timed_stage(elapsed, 'wall'):
                        output = run(self.runtime.bench_command(path, self.config.data["evaluation"]["benchmark_repetitions"]))
                finally:
                    result.engine_process_seconds.append(elapsed['wall'])
                (self.output_dir / f"{result.model}-{result.preset}-bench-{repetition}.md").write_text(output)
                samples.append(parse_bench(output))
                match = re.search(r'load_ms=([0-9.]+)', output)
                if match:
                    loads.append(float(match[1]))
            result.load_ms = median(loads) if loads else None
            p50s = [sample[0] for sample in samples if sample[0] is not None]
            p90s = [sample[1] for sample in samples if sample[1] is not None]
            result.p50_ms = median(p50s) if p50s else None
            result.p90_ms = median(p90s) if p90s else None
            if len(p50s) != 3 or len(p90s) != 3:
                raise RuntimeError("Missing latency measurements from an engine repetition")
            result.status = "engine_benchmarked"
        except Exception as exc:  # retain an inspectable partial result
            result.status, result.error = "failed", str(exc)

    def evaluation_command(self, result: Result, task_id: int, seed: int, episodes: int, port: int) -> list[str]:
        model = self.config.model(result.model)
        return [
            str(self.runtime.libero_python), str(self.runtime.source_dir / "eval/client/run_sim_client_direct.py"),
            "--task", self.config.data["evaluation"]["task"], "--task-id", str(task_id),
            "--seed", str(seed), "--n-episodes", str(episodes),
            "--arch", model["arch"], "--n-action-steps", str(model["action_steps"]),
            "--vla-addr", f"tcp://localhost:{port}", "--output-dir", str(self.output_dir / "rollouts" / result.model / result.preset),
        ]

    def _run_evaluation(self, result: Result, jobs: list[tuple[int, int]], smoke: bool) -> bool:
        artifact = artifact_path(self.config, result.model, result.preset)
        port = 5555
        server = subprocess.Popen([str(self.runtime.server), str(artifact)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(3)
        try:
            if server.poll() is not None:
                output = server.stdout.read() if server.stdout else ""
                raise RuntimeError(f"vla-server exited during cold load: {output}")
            result.cold_load_ok = True
            try:
                rss_kb = int(run(["ps", "-o", "rss=", "-p", str(server.pid)]).strip())
                result.peak_rss_mb = round(rss_kb / 1024, 1)
            except Exception:
                result.peak_rss_mb = None
            for task_id, seed in jobs:
                output = run(self.evaluation_command(result, task_id, seed, 1, port), cwd=self.runtime.source_dir, timeout=self.config.data["evaluation"]["timeout_seconds"])
                log = self.output_dir / "logs" / result.model / result.preset / f"task-{task_id}-seed-{seed}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(output)
                match = re.search(r"Success rate:\s+([0-9.]+)%\s+\(([0-9]+)/([0-9]+)\)", output)
                if not match:
                    raise RuntimeError("Could not parse LIBERO success result")
                result.successes += int(match.group(2))
                if smoke:
                    result.smoke_episodes += 1
                else:
                    result.rollout_episodes += 1
            return True
        except Exception as exc:
            result.status, result.error = "failed", str(exc)
            return False
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()

    def execute(self) -> Path:
        results = self.load_results()
        evaluation = self.config.data["evaluation"]
        for result in results:
            if result.status == "planned":
                self.collect_engine_metrics(result)
                self.save(results)
            if result.status != "engine_benchmarked":
                continue
            smoke_jobs = [(task_id, evaluation["smoke_seed"]) for task_id in evaluation["smoke_task_ids"]]
            if not self._run_evaluation(result, smoke_jobs, smoke=True):
                self.save(results)
                continue
            result.successes = 0  # smoke success is a gate, not part of the published rate
            result.rollout_episodes = 0
            jobs = [(task_id, seed) for task_id in evaluation["task_ids"] for seed in evaluation["seeds"]]
            self._run_evaluation(result, jobs, smoke=False)
            if result.status != "failed":
                result.status = "complete"
            self.save(results)
        return self.result_path()
