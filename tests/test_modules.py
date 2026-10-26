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
    "JENSEN_TEACHING_CONTROL_TOKEN_FILE",
    "FIREBIRD_TEACHING_CONTROL_TOKEN_FILE",
    "JENSEN_DECISION_ACCEPT_LICENSE",
    "FIREBIRD_DECISION_ACCEPT_LICENSE",
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


def pair(path):
    path.write_text(
        json.dumps({"url": "http://127.0.0.1:8765", "token": "a" * 32, "label": "USB computer"})
    )
    path.chmod(0o600)


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
    pair(tmp_path / "robot-lab-connection.json")
    gcp = {"project_id": "robot-project", "region": "us-central1"}
    (tmp_path / "cloud-connections.json").write_text(
        json.dumps({"version": 1, "providers": {"gcp": gcp}})
    )
    from vla_platform.module_installations import ModuleInstallations, RuntimePackage

    monkeypatch.setattr(
        ModuleInstallations,
        "inspect",
        lambda self, ident: RuntimePackage(
            id=ident,
            name=ident,
            ready=True,
            installable=True,
            message="fixture",
            minimum_free_gib=11,
        ),
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    settings = Settings(data_dir=tmp_path, cloud_runs_dir=runs)
    assert all(installed(settings).values())


def test_gemini_key_or_recording_config_is_enough(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    recording = tmp_path / "recording.json"
    recording.write_text("{}")
    result = installed(Settings(data_dir=tmp_path, recording_config=recording))
    assert result["augmentation"] and result["teaching"]
    assert not result["robot-lab"]


def test_robot_lab_config_location_follows_environment(tmp_path, monkeypatch):
    config = tmp_path / "elsewhere.json"
    pair(config)
    monkeypatch.setenv("JENSEN_ROBOT_LAB_CONFIG", str(config))
    assert installed(Settings(data_dir=tmp_path / "data"))["robot-lab"]


def test_operator_list_overrides_detection(tmp_path, monkeypatch):
    pair(tmp_path / "robot-lab-connection.json")
    monkeypatch.setenv("JENSEN_MODULES", "teaching, cloud-runs")
    result = installed(Settings(data_dir=tmp_path))
    assert [key for key, value in result.items() if value] == ["teaching", "cloud-runs"]
    monkeypatch.setenv("JENSEN_MODULES", "all")
    result = installed(Settings(data_dir=tmp_path))
    assert result["robot-lab"] and result["teaching"] and result["cloud-runs"]
    assert not result["augmentation"] and not result["decision-lab"]
    monkeypatch.setenv("JENSEN_MODULES", "")
    assert not any(installed(Settings(data_dir=tmp_path)).values())


def test_saved_enable_cannot_fabricate_missing_module(tmp_path):
    (tmp_path / "workspace-modules.json").write_text(
        json.dumps({"robot-lab": True, "decision-lab": True})
    )
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        modules = {item["id"]: item for item in client.get("/api/v1/modules").json()}
        assert modules["robot-lab"]["enabled"]
        assert not modules["robot-lab"]["available"]
        assert not modules["robot-lab"]["installed"]
        assert not modules["decision-lab"]["installed"]
        assert client.put("/api/v1/modules/robot-lab", json={"enabled": True}).status_code == 409
        assert client.put("/api/v1/modules/robot-lab", json={"enabled": False}).status_code == 200


@pytest.mark.parametrize("invalid", ["json", "permissions", "symlink"])
def test_invalid_pairing_does_not_install_module(tmp_path, invalid):
    config = tmp_path / "robot-lab-connection.json"
    pair(config)
    if invalid == "json":
        config.write_text("{}")
    elif invalid == "permissions":
        config.chmod(0o644)
    else:
        config.rename(tmp_path / "target.json")
        config.symlink_to(tmp_path / "target.json")
    assert not installed(Settings(data_dir=tmp_path))["robot-lab"]


def test_hidden_installed_module_keeps_actual_availability(tmp_path):
    pair(tmp_path / "robot-lab-connection.json")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.put("/api/v1/modules/robot-lab", json={"enabled": False}).status_code == 200
        module = next(
            item for item in client.get("/api/v1/modules").json() if item["id"] == "robot-lab"
        )
        assert module["available"] and module["configured"]
        assert not module["enabled"] and not module["installed"]


def test_incomplete_external_configuration_does_not_install_module(tmp_path, monkeypatch):
    monkeypatch.setenv("JENSEN_DECISION_ROOT", str(tmp_path))
    monkeypatch.setenv("JENSEN_TEACHING_URL", "http://127.0.0.1:9000")
    result = installed(Settings(data_dir=tmp_path, recording_config=tmp_path / "missing.json"))
    assert not result["decision-lab"] and not result["teaching"]


def test_invalid_optional_cloud_configuration_keeps_modules_available(tmp_path):
    (tmp_path / "cloud-connections.json").write_text("broken JSON")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        response = client.get("/api/v1/modules")
        assert response.status_code == 200
        module = next(item for item in response.json() if item["id"] == "augmentation")
        assert not module["available"]
