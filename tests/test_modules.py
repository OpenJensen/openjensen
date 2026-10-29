"""Feature registration, visibility, migration and runtime setup remain distinct."""

import json
import stat
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from vla_platform.api import create_app
from vla_platform.modules import (
    DEFAULT_MODULES,
    LEGACY_CORE_MODULES,
    REGISTRY_FILE,
    initialize_modules,
    module_installed,
    module_workspace_exists,
    read_registry,
    runtime_feature_readiness,
    set_module_installed,
    workspace_modules,
)
from vla_platform.settings import Settings

MODULE_ENV = (
    "JENSEN_MODULES",
    "FIREBIRD_MODULES",
    "JENSEN_ROBOT_LAB_CONFIG",
    "JENSEN_TEACHING_URL",
    "FIREBIRD_TEACHING_URL",
    "JENSEN_TEACHING_CONTROL_TOKEN_FILE",
    "FIREBIRD_TEACHING_CONTROL_TOKEN_FILE",
    "JENSEN_DECISION_ACCEPT_LICENSE",
    "FIREBIRD_DECISION_ACCEPT_LICENSE",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
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
        json.dumps(
            {
                "url": "http://127.0.0.1:8765",
                "token": "a" * 32,
                "label": "USB computer",
            }
        )
    )
    path.chmod(0o600)


def modules(client):
    response = client.get("/api/v1/modules")
    assert response.status_code == 200, response.text
    return {item["id"]: item for item in response.json()}


def test_fresh_workspace_persists_small_default_installation(tmp_path):
    assert not module_workspace_exists(tmp_path)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        values = modules(client)
        assert {key for key, item in values.items() if item["installed"]} == DEFAULT_MODULES
        assert {key for key, item in values.items() if item["enabled"]} == DEFAULT_MODULES
        assert {key for key, item in values.items() if item["default_installed"]} == DEFAULT_MODULES
        assert all(item["available"] for item in values.values())
    saved = read_registry(tmp_path)
    assert saved.version == 1 and saved.origin == "fresh"
    assert stat.S_IMODE((tmp_path / REGISTRY_FILE).stat().st_mode) == 0o600
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert {
            key for key, item in modules(client).items() if item["installed"]
        } == DEFAULT_MODULES


def test_every_feature_has_a_group(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        values = modules(client)
    assert {item["group"] for item in values.values()} == {
        "Robot Lab",
        "Data",
        "Train",
        "Test",
        "Workspace",
        "Cloud & APIs",
    }
    assert {key for key, item in values.items() if item["group"] == "Train"} == {
        "training",
        "distillation",
        "quantization",
    }
    assert {key for key, item in values.items() if item["group"] == "Cloud & APIs"} == {
        "google-cloud",
        "openai",
        "gemini",
        "huggingface",
    }


def test_install_hide_uninstall_and_runtime_setup_are_independent(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.put("/api/v1/modules/training", json={"enabled": True}).status_code == 409
        assert client.post("/api/v1/modules/training/install").status_code == 200
        training = modules(client)["training"]
        assert training["installed"] and training["enabled"] and not training["configured"]
        assert not (tmp_path / "module-environments").exists()
        assert client.put("/api/v1/modules/training", json={"enabled": False}).status_code == 200
        training = modules(client)["training"]
        assert training["installed"] and not training["enabled"]
    saved_data = tmp_path / "training-output"
    saved_data.write_text("retain user output")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert modules(client)["training"]["installed"]
        assert not modules(client)["training"]["enabled"]
        assert client.delete("/api/v1/modules/training").status_code == 200
        assert not modules(client)["training"]["installed"]
        assert client.put("/api/v1/modules/training", json={"enabled": True}).status_code == 409
        assert client.post("/api/v1/modules/training/install").status_code == 200
        assert modules(client)["training"]["enabled"]
    assert saved_data.read_text() == "retain user output"


def test_fresh_cloud_integrations_stay_uninstalled_even_with_environment_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "example-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "example-openai-key")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        values = modules(client)
        assert values["gemini"]["configured"] and values["openai"]["configured"]
        assert not any(
            values[key]["installed"]
            for key in (
                "google-cloud",
                "openai",
                "gemini",
                "huggingface",
                "augmentation",
            )
        )
        assert client.post("/api/v1/modules/google-cloud/install").status_code == 200
        assert modules(client)["google-cloud"]["installed"]
        assert not modules(client)["google-cloud"]["configured"]


def test_existing_workspace_migrates_visible_features_and_retains_hidden_choices(tmp_path):
    (tmp_path / "workspace.sqlite3").touch()
    pair(tmp_path / "robot-lab-connection.json")
    legacy = json.dumps({"robot-lab": False, "decision-lab": True})
    (tmp_path / "workspace-modules.json").write_text(legacy)
    (tmp_path / "cloud-connections.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {"gcp": {"project_id": "robot-project", "region": "us-central1"}},
            }
        )
    )
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        values = modules(client)
        assert all(values[key]["installed"] for key in LEGACY_CORE_MODULES)
        assert values["robot-lab"]["installed"] and not values["robot-lab"]["enabled"]
        assert values["robot-lab"]["configured"]
        assert values["teaching"]["installed"]
        assert values["decision-lab"]["installed"] and not values["decision-lab"]["configured"]
        assert values["google-cloud"]["installed"] and values["augmentation"]["installed"]
    assert read_registry(tmp_path).origin == "migrated"
    assert (tmp_path / "workspace-modules.json").read_text() == legacy


def test_existing_installed_runtime_is_migrated_without_configuration(tmp_path, monkeypatch):
    from vla_platform.module_installations import ModuleInstallations, RuntimePackage

    (tmp_path / "workspace.sqlite3").touch()
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
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        values = modules(client)
        assert values["robot-lab"]["installed"] and values["decision-lab"]["installed"]
        assert not values["robot-lab"]["configured"] and not values["decision-lab"]["configured"]


@pytest.mark.parametrize("invalid", ["json", "permissions", "symlink"])
def test_invalid_pairing_is_not_migrated_as_configured(tmp_path, invalid):
    config = tmp_path / "robot-lab-connection.json"
    pair(config)
    if invalid == "json":
        config.write_text("{}")
    elif invalid == "permissions":
        config.chmod(0o644)
    else:
        config.rename(tmp_path / "target.json")
        config.symlink_to(tmp_path / "target.json")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        robot = modules(client)["robot-lab"]
        assert robot["available"] and not robot["configured"] and not robot["installed"]


def test_uninstall_preserves_configuration_credentials_and_snapshot_data(tmp_path):
    pair(tmp_path / "robot-lab-connection.json")
    snapshot = tmp_path / "cloud-runs"
    snapshot.mkdir()
    (snapshot / "saved.json").write_text("{}")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        for key in ("robot-lab", "cloud-runs"):
            assert modules(client)[key]["installed"]
            assert client.delete("/api/v1/modules/" + key).status_code == 200
            assert not modules(client)[key]["installed"]
    assert (tmp_path / "robot-lab-connection.json").is_file()
    assert (snapshot / "saved.json").read_text() == "{}"
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert not modules(client)["robot-lab"]["installed"]
        assert not modules(client)["cloud-runs"]["installed"]


def test_operator_visibility_cannot_register_features(tmp_path, monkeypatch):
    monkeypatch.setenv("JENSEN_MODULES", "all")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        values = modules(client)
        assert all(item["managed"] for item in values.values())
        assert {key for key, item in values.items() if item["installed"]} == DEFAULT_MODULES
        assert client.post("/api/v1/modules/teaching/install").status_code == 409
        assert client.delete("/api/v1/modules/datasets").status_code == 409
        assert client.put("/api/v1/modules/models", json={"enabled": False}).status_code == 409
    monkeypatch.setenv("JENSEN_MODULES", "")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        values = modules(client)
        assert not any(item["enabled"] for item in values.values())
        assert {key for key, item in values.items() if item["installed"]} == DEFAULT_MODULES


@pytest.mark.parametrize("invalid", ["json", "permissions", "symlink", "version", "membership"])
def test_corrupt_registry_is_preserved_and_core_keeps_working(tmp_path, invalid):
    registry = tmp_path / REGISTRY_FILE
    valid = {"version": 1, "origin": "fresh", "installed": ["models"], "enabled": {"models": True}}
    if invalid == "version":
        valid["version"] = 2
    elif invalid == "membership":
        valid["enabled"] = {"models": True, "training": True}
    registry.write_text("invalid" if invalid == "json" else json.dumps(valid))
    registry.chmod(0o644 if invalid == "permissions" else 0o600)
    if invalid == "symlink":
        registry.rename(tmp_path / "target.json")
        registry.symlink_to(tmp_path / "target.json")
    saved = registry.read_text()
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/modules").status_code == 409
        assert client.post("/api/v1/modules/models/install").status_code == 409
    assert registry.read_text() == saved


def test_invalid_optional_cloud_configuration_does_not_break_module_catalog(tmp_path):
    (tmp_path / "cloud-connections.json").write_text("broken JSON")
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        values = modules(client)
        assert not values["google-cloud"]["configured"]
        assert not values["augmentation"]["configured"]


def test_module_installed_reads_defaults_without_writing(tmp_path):
    assert module_installed(tmp_path, "models")
    assert not module_installed(tmp_path, "openai")
    assert not list(tmp_path.iterdir())


def test_credential_metadata_controls_readiness_without_installing_integration(tmp_path):
    settings = Settings(data_dir=tmp_path)
    initialize_modules(settings, tmp_path / "robot.json", existing=False)
    values = {
        item.id: item
        for item in workspace_modules(
            settings,
            tmp_path / "robot.json",
            credentials={"openai": True, "gemini": True},
        )
    }
    assert values["openai"].configured and values["gemini"].configured
    assert not values["augmentation"].configured
    assert not values["openai"].installed and not values["gemini"].installed
    set_module_installed(tmp_path, "gemini", True)
    values = {
        item.id: item
        for item in workspace_modules(
            settings,
            tmp_path / "robot.json",
            credentials={"openai": True, "gemini": True},
        )
    }
    assert values["augmentation"].configured


def test_existing_worker_paths_control_feature_setup_readiness(tmp_path):
    from vla_platform.lifecycle.runtime import Runtime, RuntimeCatalog

    worker = tmp_path / "worker"
    worker.mkdir()
    catalog = RuntimeCatalog(
        runtimes=[
            Runtime(
                id="trainer",
                label="Trainer",
                training_only=True,
                device="cuda",
                training_python=sys.executable,
                training_root=str(worker),
            ),
            Runtime(
                id="engine",
                label="Engine",
                python=sys.executable,
                worker_root=str(worker),
                vendor="test",
                build="test",
            ),
        ]
    )
    readiness = runtime_feature_readiness(catalog)
    assert readiness["training"] and readiness["quantization"]
    assert readiness["evaluation"] and readiness["simulation"]
    assert not readiness["distillation"]
    worker.rmdir()
    assert not any(runtime_feature_readiness(catalog).values())
    assert not any(runtime_feature_readiness(RuntimeCatalog()).values())


def test_uninstall_preserves_access_to_running_dependency_install(tmp_path, monkeypatch):
    from vla_platform.module_installations import InstallOperation, ModuleInstallations

    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        assert client.post("/api/v1/modules/robot-lab/install").status_code == 200
        app.state.module_installations.operation = InstallOperation(
            id="a" * 32,
            profile="robot",
            status="running",
            stage="Installing",
            message="fixture",
            started_at="fixture",
        )
        monkeypatch.setattr(ModuleInstallations, "pending", property(lambda self: True))
        assert client.delete("/api/v1/modules/robot-lab").status_code == 409
        assert modules(client)["robot-lab"]["installed"]
        app.state.module_installations.operation = None


def test_uninstall_preserves_access_to_running_training_setup(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))

    async def close():
        return None

    with TestClient(app) as client:
        assert client.post("/api/v1/modules/training/install").status_code == 200
        app.state.execution.lifecycle.local_worker_setup = SimpleNamespace(
            state=SimpleNamespace(status="running"),
            lock=SimpleNamespace(locked=lambda: False),
            close=close,
        )
        assert client.delete("/api/v1/modules/training").status_code == 409
        assert modules(client)["training"]["installed"]


def test_uninstall_rejects_matching_active_jobs_and_allows_unrelated_feature(tmp_path):
    from vla_platform.storage import jobs

    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        assert client.post("/api/v1/modules/training/install").status_code == 200
        project = client.post("/api/v1/projects", json={"name": "Fixture"}).json()

        async def insert_job():
            async with app.state.execution.storage.engine.begin() as connection:
                await connection.execute(
                    jobs.insert().values(
                        id="active-training",
                        project_id=project["id"],
                        status="running",
                        record={"kind": "policy.finetune"},
                    )
                )

        async def remove_fixture():
            async with app.state.execution.storage.engine.begin() as connection:
                await connection.execute(jobs.delete().where(jobs.c.id == "active-training"))

        client.portal.call(insert_job)
        try:
            assert client.delete("/api/v1/modules/training").status_code == 409
            assert client.delete("/api/v1/modules/models").status_code == 200
        finally:
            client.portal.call(remove_fixture)


def test_real_to_sim_is_an_independent_data_module(tmp_path, monkeypatch):
    from vla_platform.scenes.service import Scenes

    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-openai-credential")
    monkeypatch.setattr(Scenes, "options", lambda self: {"configured": True})
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        before = modules(client)
        assert before["real-to-sim"]["group"] == "Data"
        assert before["real-to-sim"]["configured"]
        assert not before["real-to-sim"]["installed"]
        assert client.post("/api/v1/modules/real-to-sim/install").status_code == 200
        after = modules(client)
        assert after["real-to-sim"]["installed"] and after["real-to-sim"]["enabled"]
        assert after["simulation"] == before["simulation"]
        assert client.put("/api/v1/modules/real-to-sim", json={"enabled": False}).status_code == 200
        assert not modules(client)["real-to-sim"]["enabled"]
        assert client.get("/api/v1/scene-generation/options").status_code == 200
