"""Finite acceptance runner mechanics, with fixtures and no vendor or provider calls."""

import json
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import SimulationFixture, settings

from jensen_teaching import agent_acceptance as acceptance


def scenario(**changes):
    return acceptance.Scenario.parse(
        {
            "schema_version": 1,
            "task": "Pick up the configured object",
            "clock_policy": "stepped",
            "seeds": [17, 29],
            "max_steps": 20,
            "max_turns": 3,
            "max_seconds": 5,
            "max_output_tokens": 64,
            "max_total_tokens": 1024,
            "object_xy_jitter_m": 0.005,
            "evaluation": {
                "task": "lift_hold",
                "object_prim": "/World/Props/Cup",
                "support_height_m": 0.2,
                "lift_clearance_m": 0.01,
                "lift_hold_seconds": 0.1,
                "settle_seconds": 0.1,
                "maximum_linear_speed_m_s": 0.02,
                "maximum_angular_speed_rad_s": 0.1,
                "placement_min_m": None,
                "placement_max_m": None,
            },
            "scripted_actions": [{"joint": "joint_0", "delta_rad": 0.03}],
            **changes,
        }
    )


def test_fixture_smoke_does_not_claim_native_or_task_success(tmp_path):
    cfg = settings()
    spec = scenario()
    output = tmp_path / "capture"
    result = acceptance.run_episode(
        cfg,
        SimulationFixture(cfg.sim),
        output,
        spec,
        acceptance.ScriptedPlanner(spec.scripted_actions),
    )
    assert result["status"] == "fixture_executed"
    assert result["native_execution_verified"] is False
    assert result["task_success_verified"] is False and result["geometric_success"] is False
    assert result["receipts"][0]["execution"]["status"] == "completed"
    assert result["simulated_seconds"] == pytest.approx(1 / 30)
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "fixture_executed"


def test_task_planner_gets_actual_manifest_and_segment_completes_all_local_ticks(tmp_path):
    cfg = settings()
    sim = SimulationFixture(cfg.sim)
    requests = []

    def planner(request, _cancel):
        requests.append(request)
        assert threading.current_thread() is not threading.main_thread()
        assert request["action_space"]["joint_limits"]["joint_0"] == [-0.2, 0.2]
        assert request["observation"]["scene_frozen_for_planning"] is True
        if len(requests) == 1:
            return {
                "status": "proposed",
                "model": "fixture-v1",
                "skill": "joint_segment",
                "arguments": {"targets": {"joint_0": 0.15}, "duration_seconds": 0.1},
                "usage": {"total_tokens": 70},
            }
        assert request["completed_steps"] == 1
        return {
            "status": "assessed",
            "model": "fixture-v1",
            "assessment": {"outcome": "achieved", "summary": "Model claim only"},
            "usage": {"total_tokens": 4},
        }

    report = acceptance.run_episode(cfg, sim, tmp_path / "capture", scenario(), planner)
    assert report["receipts"][0]["execution"]["executed_ticks"] == 3
    assert len(sim.applications) == 3 and sim.state[0] == pytest.approx(0.15)
    assert report["token_usage_reported"] == 74 and report["token_budget_charged"] == 134
    assert report["model_assessment"]["outcome"] == "achieved"
    assert report["task_success_verified"] is False


def test_step_and_token_limits_stop_before_admitting_another_action(tmp_path):
    cfg = settings()
    sim = SimulationFixture(cfg.sim)
    spec = scenario(max_total_tokens=64)
    result = acceptance.run_episode(
        cfg, sim, tmp_path / "tokens", spec, acceptance.ScriptedPlanner(spec.scripted_actions)
    )
    assert result["termination"] == "token_budget_exhausted"
    assert len(sim.applications) == 1
    other = SimulationFixture(cfg.sim)

    def segment(_request, _cancel):
        return {
            "status": "proposed",
            "model": "fixture-v1",
            "skill": "joint_segment",
            "arguments": {"targets": {"joint_0": 0.1}, "duration_seconds": 0.2},
        }

    result = acceptance.run_episode(cfg, other, tmp_path / "steps", scenario(max_steps=2), segment)
    assert result["termination"] == "step_budget_exhausted_before_admission"
    assert not other.applications


def test_continuous_clock_advances_while_planning_but_frozen_clock_does_not(tmp_path):
    cfg = settings()
    for clock in ("continuous", "stepped"):
        sim = SimulationFixture(cfg.sim)

        def planner(_request, _cancel):
            time.sleep(0.08)
            return {
                "status": "assessed",
                "model": "fixture-v1",
                "assessment": {"outcome": "needs_help", "summary": "No action"},
            }

        report = acceptance.run_episode(
            cfg, sim, tmp_path / clock, scenario(clock_policy=clock), planner
        )
        assert report["status"] == "fixture_executed"
        assert (sim.counter > 0) is (clock == "continuous")
        assert report["planning_wall_seconds"] >= 0.08
        assert report["native_execution_verified"] is False


def test_preflight_never_accepts_license_or_imports_kit(monkeypatch):
    monkeypatch.delenv("ACCEPT_EULA", raising=False)
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    monkeypatch.setattr(acceptance, "hardware_preflight", lambda: {"reasons": []})
    monkeypatch.setattr(acceptance.importlib.metadata, "version", lambda _name: "6.1.0.0")
    monkeypatch.setattr(acceptance.importlib.util, "find_spec", lambda _name: object())
    result = acceptance.runtime_preflight()
    assert result["status"] == "blocked" and result["native_execution_verified"] is False
    assert any("license acceptance" in reason for reason in result["reasons"])
    assert "ACCEPT_EULA" not in acceptance.os.environ
    assert "OMNI_KIT_ACCEPT_EULA" not in acceptance.os.environ


def test_scenario_rejects_unbounded_or_unpaired_settings():
    for changes in (
        {"seeds": [1, 1]},
        {"max_turns": 61},
        {"max_seconds": True},
        {"object_xy_jitter_m": 0.04},
        {"max_total_tokens": 63},
    ):
        with pytest.raises(ValueError):
            scenario(**changes)


def test_isolated_provider_protocol_accepts_abstract_task_without_claiming_pickup(tmp_path):
    cfg = settings()
    source = Path(__file__).resolve().parents[3] / "workers/agent_control/src"
    planner = acceptance.ProcessPlanner(sys.executable, "mock", "mock-v1", "low", source)
    sim = SimulationFixture(cfg.sim)
    result = acceptance.run_episode(cfg, sim, tmp_path / "capture", scenario(), planner)
    assert result["status"] == "fixture_executed"
    assert result["model_assessment"]["outcome"] == "needs_help"
    assert result["task_success_verified"] is False and not sim.applications
