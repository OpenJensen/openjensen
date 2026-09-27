"""Invalid temporal requests cannot allocate a persisted job or worker."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from test_training_catalog import runtime
from vla_platform.api import create_app
from vla_platform.lifecycle.runtime import RuntimeCatalog
from vla_platform.lifecycle.training_catalog import TRAINING_MODEL_BY_ID
from vla_platform.settings import Settings


@pytest.mark.parametrize(
    "family,recipe,match",
    [
        ("diffusion", {"chunk_size": 13}, "does not support chunk_size"),
        ("multi_task_dit", {"chunk_size": 13}, "does not support chunk_size"),
        ("vqbet", {"chunk_size": 13}, "does not support chunk_size"),
        ("act", {"prediction_horizon": 3, "execution_horizon": 4}, "no greater"),
        ("act", {"prediction_horizon": True}, "integer"),
        ("smolvla", {"frame_stride": 2}, "frame_stride=1"),
        ("psi0", {"chunk_size": 29}, "chunk_size=30"),
        ("pi05", {"execution_horizon": 2}, "not verified"),
    ],
)
def test_api_temporal_failure_precedes_job_allocation(tmp_path, monkeypatch, family, recipe, match):
    config = tmp_path / "runtimes.json"
    config.write_text(
        RuntimeCatalog(
            runtimes=[
                runtime(
                    training_module="custom_training.application",
                    training_model_ids=[family],
                )
            ]
        ).model_dump_json()
    )
    app = create_app(Settings(data_dir=tmp_path / "data", runtime_config=config))
    with TestClient(app) as client:
        client.put("/api/v1/compute-settings", json={"local": {"enabled": True}})
        pid = client.post("/api/v1/projects", json={"name": "Temporal preflight"}).json()["id"]
        before = client.get(f"/api/v1/projects/{pid}/jobs").json()
        create = AsyncMock(side_effect=AssertionError("invalid recipe allocated a job"))
        # The API's validate path must reject before the owner persists or starts work.
        monkeypatch.setattr(app.state.execution, "run", create)
        model = TRAINING_MODEL_BY_ID[family]
        response = client.post(
            f"/api/v1/projects/{pid}/policy-jobs",
            json={
                "operation": "policy.finetune",
                "runtime_id": "trainer",
                "dataset_job_id": "absent",
                "training_method": "qlora" if family == "smolvla" else "full",
                "training": {
                    "model_id": model.model_id,
                    "model_revision": model.model_revision,
                    **recipe,
                },
            },
        )
        assert response.status_code == 422, response.text
        assert match in response.json()["detail"]
        assert client.get(f"/api/v1/projects/{pid}/jobs").json() == before
        create.assert_not_awaited()
