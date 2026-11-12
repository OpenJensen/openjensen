"""Self-service native ACT artifact installation: exact bytes, admission and cleanup."""

import asyncio
import base64
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError
from vla_platform.robot_lab_bridge import RobotLabBridge, RobotLabError
from vla_platform.robot_policy_install import (
    JOINTS,
    UNITS,
    PolicyInstallRequest,
    _descriptor,
    install_artifact,
)

TOKEN = "test-private-policy-install-" + "a" * 32


def bundle(tmp_path, *, known=True, corrupt=False):
    metadata = {
        "architecture": "act",
        "training_backend": "lerobot",
        "reload_verified": True,
        "camera_keys": ["observation.images.front"],
    }
    if known:
        metadata.update(fps=13, state_units=UNITS, action_units=UNITS, joint_names=JOINTS)
    contents = {"checkpoint/pretrained_model/config.json": b'{"type":"act"}'}
    files = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    raw = json.dumps({"schema_version": 1, "metadata": metadata, "files": files}).encode()
    archive = tmp_path / "checkpoint.tar"
    with tarfile.open(archive, "w") as stream:
        for name, content in {"manifest.json": raw, **contents}.items():
            if corrupt and name != "manifest.json":
                content += b"changed"
            item = tarfile.TarInfo("policy/" + name)
            item.size = len(content)
            stream.addfile(item, io.BytesIO(content))
    artifact = SimpleNamespace(
        id="job:checkpoint-000100",
        project_id="project",
        format="training_checkpoint",
        metadata=metadata,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        file_bytes=sum(len(content) for content in contents.values()),
    )
    return archive, artifact


class Download:
    def __init__(self, content, fail=False):
        self.content, self.fail, self.closed = content, fail, False

    async def __aiter__(self):
        yield self.content[:1024]
        if self.fail:
            raise RuntimeError("cloud integrity check failed")
        yield self.content[1024:]

    async def aclose(self):
        self.closed = True


def application(tmp_path, archive, artifact, handler, source=None):
    class Projects:
        async def get(self, project):
            return {"id": project} if project == "project" else None

    class Lifecycle:
        downloads = 0

        async def artifact(self, project, ident):
            assert project == "project" and ident == artifact.id
            return artifact

        async def download(self, project, ident):
            self.downloads += 1
            return source if source is not None else archive

    data = tmp_path / "app"
    data.mkdir()
    connection = data / "robot-lab-connection.json"
    connection.write_text(json.dumps({"url": "http://127.0.0.1:18101", "token": TOKEN}))
    connection.chmod(0o600)
    bridge = RobotLabBridge(data, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    app = SimpleNamespace(
        state=SimpleNamespace(
            projects=Projects(), execution=SimpleNamespace(lifecycle=Lifecycle()), robot_lab=bridge
        )
    )
    return app


@pytest.fixture(autouse=True)
def disk_headroom(monkeypatch):
    monkeypatch.setattr(
        "vla_platform.robot_policy_install.shutil.disk_usage",
        lambda _: SimpleNamespace(free=100 * 1024**3),
    )


def test_install_streams_exact_verified_archive_and_never_starts_control(tmp_path):
    archive, artifact = bundle(tmp_path)
    calls = []

    async def handler(request):
        calls.append(request.url.path)
        assert request.headers["authorization"] == "Bearer " + TOKEN
        assert request.headers["content-type"] == "application/x-tar"
        assert request.headers.get("transfer-encoding") is None
        assert int(request.headers["content-length"]) == archive.stat().st_size
        assert await request.aread() == archive.read_bytes()
        header = json.loads(base64.b64decode(request.headers["x-openjensen-policy"], validate=True))
        assert header["fps"] == 13 and header["state_units"] == UNITS
        assert header["joint_names"] == JOINTS
        assert header["source_manifest_sha256"] == artifact.manifest_sha256
        assert header["id"].startswith("artifact-") and header["name"] == "Pen policy"
        return httpx.Response(200, json={"policies": [], "token": TOKEN, "message": TOKEN})

    app = application(tmp_path, archive, artifact, handler)

    async def run():
        result = await install_artifact(
            app, "project", artifact.id, PolicyInstallRequest(name="Pen policy")
        )
        assert result == {"policies": [], "message": "[private]"}
        await app.state.robot_lab.close()

    asyncio.run(run())
    assert calls == ["/v1/policy/install"]
    assert not list((app.state.robot_lab.data_dir / "policy-install").iterdir())


def test_unknown_coordinates_require_explicit_units_and_training_fps(tmp_path):
    _, artifact = bundle(tmp_path, known=False)
    for payload in (
        PolicyInstallRequest(name="Pen"),
        PolicyInstallRequest(name="Pen", fps=13),
        PolicyInstallRequest(name="Pen", confirm_so101_units=True),
    ):
        with pytest.raises(ValueError):
            _descriptor(artifact, artifact.metadata, {}, payload)
    descriptor = _descriptor(
        artifact,
        artifact.metadata,
        {},
        PolicyInstallRequest(name="Pen", fps=13, confirm_so101_units=True),
    )
    assert descriptor["fps"] == 13 and descriptor["action_units"] == UNITS


def test_declared_incompatible_units_or_fps_cannot_be_overridden(tmp_path):
    _, artifact = bundle(tmp_path)
    payload = PolicyInstallRequest(name="Pen", fps=10, confirm_so101_units=True)
    with pytest.raises(ValueError, match="FPS unchanged"):
        _descriptor(artifact, artifact.metadata, {}, payload)
    artifact.metadata["action_units"] = "radians"
    with pytest.raises(ValueError, match="units are incompatible"):
        _descriptor(
            artifact,
            artifact.metadata,
            {},
            PolicyInstallRequest(name="Pen", fps=13, confirm_so101_units=True),
        )


def test_bad_artifact_is_rejected_before_download_or_companion_request(tmp_path):
    archive, artifact = bundle(tmp_path)
    artifact.metadata["reload_verified"] = False
    app = application(tmp_path, archive, artifact, lambda _: pytest.fail("contacted companion"))

    async def run():
        with pytest.raises(RobotLabError) as error:
            await install_artifact(app, "project", artifact.id, PolicyInstallRequest(name="Pen"))
        assert error.value.status == 422
        assert app.state.execution.lifecycle.downloads == 0
        await app.state.robot_lab.close()

    asyncio.run(run())


def test_changed_archive_never_reaches_companion_and_spool_is_removed(tmp_path):
    archive, artifact = bundle(tmp_path, corrupt=True)
    app = application(tmp_path, archive, artifact, lambda _: pytest.fail("contacted companion"))

    async def run():
        with pytest.raises(RobotLabError):
            await install_artifact(app, "project", artifact.id, PolicyInstallRequest(name="Pen"))
        await app.state.robot_lab.close()

    asyncio.run(run())
    assert not list((app.state.robot_lab.data_dir / "policy-install").iterdir())


def test_cloud_download_finishes_and_closes_before_upload(tmp_path):
    archive, artifact = bundle(tmp_path)
    source = Download(archive.read_bytes())

    async def handler(request):
        assert source.closed
        assert await request.aread() == archive.read_bytes()
        return httpx.Response(200, json={"installed_policy_id": "example"})

    app = application(tmp_path, archive, artifact, handler, source)

    async def run():
        result = await install_artifact(
            app, "project", artifact.id, PolicyInstallRequest(name="Pen")
        )
        assert result["installed_policy_id"] == "example"
        await app.state.robot_lab.close()

    asyncio.run(run())
    assert source.closed


def test_cloud_integrity_failure_closes_source_and_never_uploads(tmp_path):
    archive, artifact = bundle(tmp_path)
    source = Download(archive.read_bytes(), fail=True)
    app = application(
        tmp_path, archive, artifact, lambda _: pytest.fail("contacted companion"), source
    )

    async def run():
        with pytest.raises(RobotLabError):
            await install_artifact(app, "project", artifact.id, PolicyInstallRequest(name="Pen"))
        await app.state.robot_lab.close()

    asyncio.run(run())
    assert source.closed
    assert not list((app.state.robot_lab.data_dir / "policy-install").iterdir())


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "Pen", "fps": True},
        {"name": "Pen", "confirm_so101_units": "yes"},
        {"name": "Pen", "fps": 41},
        {"name": "\n"},
        {"name": "Pen", "path": "/untrusted"},
    ],
)
def test_install_request_rejects_coercion_and_arbitrary_paths(payload):
    with pytest.raises(ValidationError):
        PolicyInstallRequest.model_validate(payload)


def test_trainer_temporal_fps_and_pos_joint_names_need_no_manual_fps(tmp_path):
    _, artifact = bundle(tmp_path, known=False)
    artifact.metadata.update(
        temporal_contract={"action_fps": 13},
        state_names=[name + ".pos" for name in JOINTS],
        action_names=[name + ".pos" for name in JOINTS],
    )
    descriptor = _descriptor(
        artifact,
        artifact.metadata,
        {},
        PolicyInstallRequest(name="Pen", confirm_so101_units=True),
    )
    assert descriptor["fps"] == 13 and descriptor["joint_names"] == JOINTS
    with pytest.raises(ValueError, match="Confirm six"):
        _descriptor(artifact, artifact.metadata, {}, PolicyInstallRequest(name="Pen"))
    with pytest.raises(ValueError, match="FPS unchanged"):
        _descriptor(
            artifact,
            artifact.metadata,
            {},
            PolicyInstallRequest(name="Pen", fps=10, confirm_so101_units=True),
        )


@pytest.mark.parametrize("key", ["state_names", "action_names"])
@pytest.mark.parametrize("change", ["reordered", "extra", "non_joint", "not_list"])
def test_trainer_joint_name_mismatches_are_rejected(tmp_path, key, change):
    _, artifact = bundle(tmp_path, known=False)
    names = [name + ".pos" for name in JOINTS]
    if change == "reordered":
        names[0], names[1] = names[1], names[0]
    elif change == "extra":
        names.append("extra.pos")
    elif change == "non_joint":
        names[0] = "camera.pos"
    else:
        names = "shoulder_pan.pos"
    artifact.metadata.update(temporal_contract={"action_fps": 13}, **{key: names})
    with pytest.raises(ValueError, match="joint order"):
        _descriptor(
            artifact,
            artifact.metadata,
            {},
            PolicyInstallRequest(name="Pen", confirm_so101_units=True),
        )


def test_offline_native_import_preserves_unknown_units_and_timing(tmp_path):
    _, artifact = bundle(tmp_path, known=False)
    value = _descriptor(
        artifact, artifact.metadata, {}, PolicyInstallRequest(name="Offline", offline_only=True)
    )
    assert value["offline_only"] is True
    assert value["fps"] is None and value["state_units"] is None and value["action_units"] is None
    assert value["joint_names"] is None and value["id"].endswith("-offline")


def test_smol_unknown_control_semantics_cannot_be_claimed_by_checkbox(tmp_path):
    _, artifact = bundle(tmp_path, known=False)
    artifact.metadata["architecture"] = "smolvla"
    value = _descriptor(
        artifact,
        artifact.metadata,
        {},
        PolicyInstallRequest(name="Base", fps=13, confirm_so101_units=True),
    )
    assert value["offline_only"] is True
    assert value["joint_names"] is None and value["action_units"] is None


RECORDED = {
    "units": "degrees; gripper percent",
    "state_names": [name + ".pos" for name in JOINTS],
    "action_names": [name + ".pos" for name in JOINTS],
    "dataset_fps": 23,
}


def test_a_policy_trained_on_a_robot_lab_recording_inherits_its_units(tmp_path):
    """SmolVLA may drive the arm only when units and joint order are on record. A tick-box
    is not a record; the verified recording the policy was trained on is."""
    _, artifact = bundle(tmp_path, known=False)
    artifact.metadata["architecture"] = "smolvla"
    request = PolicyInstallRequest(name="Duck")
    without = _descriptor(artifact, artifact.metadata, {}, request)
    assert without["offline_only"] is True
    value = _descriptor(artifact, artifact.metadata, {}, request, RECORDED)
    assert value["offline_only"] is False and not value["id"].endswith("-offline")
    assert value["fps"] == 23 and value["joint_names"] == JOINTS
    assert value["state_units"] == UNITS and value["action_units"] == UNITS
    # Asking for offline use is still honoured, and a different rate is still refused.
    offline = _descriptor(
        artifact,
        artifact.metadata,
        {},
        PolicyInstallRequest(name="Duck", offline_only=True),
        RECORDED,
    )
    assert offline["offline_only"] is True
    with pytest.raises(ValueError, match="saved training FPS"):
        _descriptor(
            artifact, artifact.metadata, {}, PolicyInstallRequest(name="D", fps=30), RECORDED
        )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"units": "radians"}, "units are incompatible"),
        ({"action_names": [name + ".pos" for name in reversed(JOINTS)]}, "joint order"),
        ({"dataset_fps": 120}, "FPS is unsupported"),
    ],
)
def test_a_recording_in_other_units_or_joint_order_still_blocks_arm_control(
    tmp_path, change, match
):
    _, artifact = bundle(tmp_path, known=False)
    artifact.metadata["architecture"] = "smolvla"
    with pytest.raises(ValueError, match=match):
        _descriptor(
            artifact,
            artifact.metadata,
            {},
            PolicyInstallRequest(name="Duck"),
            {**RECORDED, **change},
        )


def test_recorded_units_are_read_only_from_a_verified_robot_lab_training_copy(tmp_path):
    from test_dataset_combine import record, worker
    from test_dataset_snapshots import build_dataset
    from vla_platform.contracts import DatasetSnapshot
    from vla_platform.datasets.snapshots import create_snapshot
    from vla_platform.robot_policy_install import recorded_control

    store = tmp_path / "workspace/dataset-snapshots"
    # Two one-episode recordings, combined, make a valid training copy. Combining keeps
    # what the recorder said about units.
    two = tmp_path / "two"
    runs = record(tmp_path / "recorded", 2)
    code, _ = worker({"operation": "merge", "roots": [str(r) for r in runs], "output": str(two)})
    assert code == 0
    robot = DatasetSnapshot.model_validate(create_snapshot(tmp_path, two, store))
    imported = DatasetSnapshot.model_validate(
        create_snapshot(tmp_path, build_dataset(tmp_path / "video"), store)
    )

    def app(snapshot, *, project="project", many=False):
        jobs = {
            "train": SimpleNamespace(
                project_id="project",
                request=SimpleNamespace(
                    dataset_job_id="data", dataset_job_ids=["data", "x"] if many else None
                ),
            ),
            "data": SimpleNamespace(project_id=project, result=SimpleNamespace(snapshot=snapshot)),
        }

        async def get(ident):
            return jobs.get(ident)

        execution = SimpleNamespace(
            get=get, settings=SimpleNamespace(data_dir=tmp_path / "workspace")
        )
        return SimpleNamespace(state=SimpleNamespace(execution=execution))

    artifact = SimpleNamespace(job_id="train")
    assert asyncio.run(recorded_control(app(robot), "project", artifact)) == {
        "units": "degrees; gripper percent",
        "state_names": [name + ".pos" for name in JOINTS],
        "action_names": [name + ".pos" for name in JOINTS],
        "dataset_fps": 10,
    }
    # Not a Robot Lab recording, another project's data, or several datasets: no evidence.
    assert asyncio.run(recorded_control(app(imported), "project", artifact)) is None
    assert asyncio.run(recorded_control(app(robot, project="other"), "project", artifact)) is None
    assert asyncio.run(recorded_control(app(robot, many=True), "project", artifact)) is None
    assert asyncio.run(recorded_control(app(None), "project", artifact)) is None
