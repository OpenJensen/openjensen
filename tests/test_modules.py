"""Optional module detection reads saved configuration only; nothing is contacted."""

import json

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.settings import Settings

MODULE_ENV = (
    "JENSEN_MODULES",
    "FIREBIRD_MODULES",
    "JENSEN_ROBOT_LAB_CONFIG",
    "JENSEN_TEACHING_URL",
    "FIREBIRD_TEACHING_URL",
    "GEMINI_API_KEY",
    "JENSEN_DECISION_PYTHON",
    "JENSEN_DECISION_ROOT",
    "JENSEN_DECISION_MODEL_DIR",
    "FIREBIRD_DECISION_PYTHON",
    "FIREBIRD_DECISION_ROOT",
    "FIREBIRD_DECISION_MODEL_DIR",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in MODULE_ENV:
        monkeypatch.delenv(name, raising=False)


def installed(settings):
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/modules")
    assert response.status_code == 200
    return {item["id"]: item["installed"] for item in response.json()}


def test_fresh_workspace_has_no_optional_modules(tmp_path):
    assert installed(Settings(data_dir=tmp_path)) == {
        "robot-lab": False,
        "teaching": False,
        "augmentation": False,
        "decision-lab": False,
        "cloud-runs": False,
    }


def test_saved_configuration_installs_each_module(tmp_path, monkeypatch):
    (tmp_path / "robot-lab-connection.json").write_text("{}")
    gcp = {"project_id": "robot-project", "region": "us-central1"}
    (tmp_path / "cloud-connections.json").write_text(
        json.dumps({"version": 1, "providers": {"gcp": gcp}})
    )
    monkeypatch.setenv("FIREBIRD_TEACHING_URL", "http://127.0.0.1:9000")
    monkeypatch.setenv("JENSEN_DECISION_ROOT", str(tmp_path))
    settings = Settings(data_dir=tmp_path, cloud_runs_dir=tmp_path / "runs")
    assert all(installed(settings).values())


def test_gemini_key_or_recording_config_is_enough(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    result = installed(Settings(data_dir=tmp_path, recording_config=tmp_path / "recording.json"))
    assert result["augmentation"] and result["teaching"]
    assert not result["robot-lab"]


def test_robot_lab_config_location_follows_environment(tmp_path, monkeypatch):
    config = tmp_path / "elsewhere.json"
    config.write_text("{}")
    monkeypatch.setenv("JENSEN_ROBOT_LAB_CONFIG", str(config))
    assert installed(Settings(data_dir=tmp_path / "data"))["robot-lab"]


def test_operator_list_overrides_detection(tmp_path, monkeypatch):
    (tmp_path / "robot-lab-connection.json").write_text("{}")
    monkeypatch.setenv("JENSEN_MODULES", "teaching, cloud-runs")
    result = installed(Settings(data_dir=tmp_path))
    assert [key for key, value in result.items() if value] == ["teaching", "cloud-runs"]
    monkeypatch.setenv("JENSEN_MODULES", "all")
    assert all(installed(Settings(data_dir=tmp_path)).values())
    monkeypatch.setenv("JENSEN_MODULES", "")
    assert not any(installed(Settings(data_dir=tmp_path)).values())
