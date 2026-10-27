"""Application resource measurements without GPU access or native executables."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from policykit import measurement


def test_percentile_interpolates_between_observed_calls():
    assert measurement.percentile([40, 10, 30, 20], 50) == 25
    assert measurement.percentile([40, 10, 30, 20], 95) == pytest.approx(38.5)
    assert measurement.percentile([7], 95) == 7


@pytest.mark.parametrize("exit_code", [0, 3])
def test_measure_parses_process_peak_rss_and_always_stops_monitor(
    tmp_path, monkeypatch, exit_code
):
    monitor_calls = []
    process_calls = []
    poll_results = iter([None, exit_code, exit_code])
    process = SimpleNamespace(
        pid=765432,
        poll=lambda: next(poll_results, exit_code),
        returncode=exit_code,
    )
    monitor = SimpleNamespace(
        terminate=lambda: monitor_calls.append("terminate"),
        wait=lambda timeout: monitor_calls.append(("wait", timeout)),
    )
    read_text = Path.read_text

    def process_status(path, *args, **kwargs):
        if path == Path("/proc/765432/status"):
            return "VmRSS:\t2048 kB\nVmHWM:\t4096 kB\n"
        return read_text(path, *args, **kwargs)

    def popen(command, *, stdout, **kwargs):
        if command[0] == "nvidia-smi":
            stdout.write("2026-01-01, 150, 0, 30, 5\n")
            return monitor
        process_calls.append(command)
        return process

    monkeypatch.setattr(Path, "read_text", process_status)
    monkeypatch.setattr(measurement.subprocess, "check_output", lambda *a, **k: "100")
    monkeypatch.setattr(measurement.subprocess, "Popen", popen)
    monkeypatch.setattr(measurement.time, "sleep", lambda _: None)
    log = tmp_path / "inference.log"
    command = ["synthetic-policy-process"]
    if exit_code:
        with pytest.raises(RuntimeError, match="Exit 3"):
            measurement.measure(command, log)
    else:
        result = measurement.measure(command, log)
        assert result["sampled_peak_rss_mib"] == 4
        assert result["device_used_mib_before"] == 100
        assert result["sampled_peak_device_used_mib"] == 150
        assert result["sampled_device_increase_mib"] == 50
        assert result["gpu_samples"] == 1
        assert result["gpu_telemetry"] == str(log.with_suffix(".gpu.csv"))
    assert process_calls == [command]
    assert monitor_calls == ["terminate", ("wait", 5)]
