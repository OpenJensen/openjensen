"""A dataset or a model published on one machine is the same thing on another.

Two real applications, each with its own workspace, stand for the two machines. Between
them is ``hub_stand_in.py``: the Hugging Face client process replaced by one that keeps
"the Hub" in a local folder. Everything else is real: the saved dataset, the model
bundle, the jobs, the files on disk and the installation on the robot's companion.
Requests go through the routes in ``huggingface_sharing_api.py``.
"""

import asyncio
import base64
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert
from test_dataset_snapshots import build_dataset
from vla_platform import huggingface_sharing as sharing
from vla_platform import huggingface_transfer as transfer
from vla_platform import huggingface_transfer_worker as worker
from vla_platform.api import create_app
from vla_platform.contracts import Job, now
from vla_platform.lifecycle.contracts import LifecycleResult, PolicyArtifact, PolicyRequest
from vla_platform.robot_lab_bridge import RobotLabBridge
from vla_platform.robot_policy_install import JOINTS, UNITS, recorded_control
from vla_platform.settings import Settings
from vla_platform.storage import jobs

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Saved credentials require POSIX modes")
TOKEN = "hf_" + "w" * 30
COMPANION = "test-private-companion-" + "a" * 32
STAND_IN = Path(__file__).with_name("hub_stand_in.py")
CAMERA = "observation.images.front"


@pytest.fixture
def hub(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setenv("FAKE_HUB", str(root))
    monkeypatch.setenv("JENSEN_HUGGINGFACE_PYTHON", sys.executable)
    monkeypatch.setattr(transfer, "WORKER", STAND_IN)
    return root


def machine(tmp_path, name, token=TOKEN):
    """One computer running the application; ``token`` is what its Settings page holds."""
    data = tmp_path / name
    data.mkdir()
    if token:
        credential = data / "huggingface-credential.json"
        credential.write_text(
            json.dumps({"version": 1, "token": token, "username": "you", "checked_at": now()})
        )
        credential.chmod(0o600)
    return TestClient(create_app(Settings(data_dir=data)))


def project(client, name="Pick up the pen"):
    return client.post("/api/v1/projects", json={"name": name}).json()["id"]


def hashes(folder):
    return {
        path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def save_dataset(client, pid, folder, name):
    """Save a folder as a dataset the way the page does: begin, one file at a time, finish."""
    ident = client.post(f"/api/v1/projects/{pid}/dataset-uploads", json={"name": name}).json()["id"]
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            sent = client.put(
                f"/api/v1/dataset-uploads/{ident}/file",
                params={"path": path.relative_to(folder).as_posix()},
                content=path.read_bytes(),
            )
            assert sent.status_code == 200, sent.text
    saved = client.post(f"/api/v1/dataset-uploads/{ident}/finish")
    assert saved.status_code == 200 and saved.json()["status"] == "ready", saved.text
    return saved.json()


def finished(client, pid, job, status="succeeded"):
    deadline = time.monotonic() + 40
    while True:
        value = client.get(f"/api/v1/projects/{pid}/work-jobs/{job['id']}").json()
        if value["status"] != "running" or time.monotonic() > deadline:
            assert value["status"] == status, value
            return value
        time.sleep(0.05)


def share(client, pid, action, **body):
    response = client.post(f"/api/v1/projects/{pid}/huggingface-sharing/{action}", json=body)
    assert response.status_code == 202, response.text
    return response.json()


def calls(hub):
    return [json.loads(line) for line in (hub / "calls.jsonl").read_text().splitlines()]


def nowhere_but_the_credential(root, secret):
    """The token is in the one private credential file and in no other file at all."""
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "huggingface-credential.json"
        and secret.encode() in path.read_bytes()
    ]


def test_a_dataset_published_on_one_machine_is_the_same_dataset_on_another(hub, tmp_path):
    recording = tmp_path / "recording"
    build_dataset(recording, episodes=3)
    with machine(tmp_path, "sender") as sender, machine(tmp_path, "receiver") as receiver:
        pid = project(sender)
        original = save_dataset(sender, pid, recording, "Pick up the pen!")
        labelled = sender.put(
            f"/api/v1/datasets/{original['id']}/annotations",
            json={"revision": 0, "views": {CAMERA: "Front camera"}, "frames": {}},
        )
        assert labelled.status_code == 200, labelled.text

        # The page shows where the dataset will go, and that it will be private, before
        # anything is sent.
        destination = sender.get(
            f"/api/v1/projects/{pid}/huggingface-sharing/destination",
            params={"kind": "dataset", "id": original["id"]},
        ).json()
        assert destination["repo_id"] == "you/pick-up-the-pen" and destination["private"] is True
        assert destination["account"] == {
            "connected": True,
            "username": "you",
            "source": "settings",
            "message": None,
        }
        assert destination["published"] is None and not (hub / "calls.jsonl").exists()

        published = finished(
            sender, pid, share(sender, pid, "publish", kind="dataset", id=original["id"])
        )
        assert published["operation"] == "dataset.publish" and published["module"] == "datasets"
        link = published["result"]["url"]
        assert link == "https://huggingface.co/datasets/you/pick-up-the-pen"
        assert published["result"]["private"] is True
        assert published["outputs"] == [{"label": "Open on Hugging Face", "url": link}]
        assert [event["stage"] for event in published["events"]] == [
            "preparing",
            "uploading",
            "succeeded",
        ]
        repository = hub / "dataset/you/pick-up-the-pen"
        assert json.loads((hub / "dataset/you/pick-up-the-pen.settings.json").read_text()) == {
            "private": True
        }
        assert (hub / "dataset/you/pick-up-the-pen.tag").read_text() == "v3.0"
        assert hashes(recording).items() <= hashes(repository).items()
        assert "LeRobot" in (repository / "README.md").read_text()
        again = sender.get(
            f"/api/v1/projects/{pid}/huggingface-sharing/destination",
            params={"kind": "dataset", "id": original["id"]},
        ).json()
        assert again["published"]["url"] == link and again["published"]["private"] is True

        # The other machine has never seen this dataset: one pasted link brings it over.
        other = project(receiver, "Run on the arm")
        fetched = finished(
            receiver, other, share(receiver, other, "fetch", kind="dataset", link=link)
        )
        result = fetched["result"]
        assert result["verified"] is True and result["already_saved"] is False
        copy = receiver.get(f"/api/v1/datasets/{result['dataset_id']}").json()
        assert copy["status"] == "ready" and copy["name"] == "Pick up the pen!"
        assert copy["project_id"] == other
        assert copy["detection"]["sha256"] == original["detection"]["sha256"]
        assert copy["annotations"]["views"] == {CAMERA: "Front camera"}
        assert copy["huggingface"]["repo_id"] == "you/pick-up-the-pen"
        saved = receiver.app.state.dataset_library.resolve(result["dataset_id"], other)
        assert hashes(saved) == hashes(recording)
        listed = receiver.get("/api/v1/datasets", params={"project_id": other}).json()
        assert [row["id"] for row in listed] == [result["dataset_id"]]

        # Pasting the same link again gives the dataset that is already there.
        second = finished(
            receiver,
            other,
            share(receiver, other, "fetch", kind="dataset", link="you/pick-up-the-pen"),
        )
        assert second["result"]["dataset_id"] == result["dataset_id"]
        assert second["result"]["already_saved"] is True
        assert len(receiver.get("/api/v1/datasets", params={"project_id": other}).json()) == 1

        history = receiver.get(f"/api/v1/projects/{other}/work-jobs").json()
        assert {"dataset.fetch", "dataset.import"} == {row["operation"] for row in history}
        assert not (tmp_path / "receiver/huggingface-work").exists()

    # The token reached the client through its environment and is written nowhere.
    assert all(call["token"] == TOKEN for call in calls(hub))
    assert TOKEN not in json.dumps([[call["argv"], call["request"]] for call in calls(hub)])
    for name in ("sender", "receiver"):
        assert nowhere_but_the_credential(tmp_path / name, TOKEN) == []


def test_a_damaged_or_unfinished_repository_is_refused_and_leaves_nothing_behind(
    hub, tmp_path, monkeypatch
):
    recording = tmp_path / "recording"
    build_dataset(recording, episodes=2)
    with machine(tmp_path, "sender") as sender, machine(tmp_path, "receiver") as receiver:
        pid, other = project(sender), project(receiver)
        original = save_dataset(sender, pid, recording, "First take")

        # The connection drops after the first file: the publish fails and says so.
        monkeypatch.setenv("FAKE_HUB_STOP_AFTER_BEGIN", "you/first-take")
        failed = finished(
            sender, pid, share(sender, pid, "publish", kind="dataset", id=original["id"]), "failed"
        )
        assert "Could not reach Hugging Face" in failed["message"]
        half = finished(
            receiver,
            other,
            share(receiver, other, "fetch", kind="dataset", link="you/first-take"),
            "failed",
        )
        assert "did not finish" in half["message"]

        # Publishing again completes the same repository.
        monkeypatch.delenv("FAKE_HUB_STOP_AFTER_BEGIN")
        finished(sender, pid, share(sender, pid, "publish", kind="dataset", id=original["id"]))

        # Someone changes a file and adds another on the Hub afterwards.
        data = hub / "dataset/you/first-take/data/chunk-000/file-000.parquet"
        intact = data.read_bytes()
        data.write_bytes(intact + b"changed")
        damaged = finished(
            receiver,
            other,
            share(receiver, other, "fetch", kind="dataset", link="you/first-take"),
            "failed",
        )
        assert (
            "differs from what was published (data/chunk-000/file-000.parquet)"
            in damaged["message"]
        )
        assert receiver.get("/api/v1/datasets", params={"project_id": other}).json() == []
        assert not (tmp_path / "receiver/huggingface-work").exists()

        # An extra file that was never published is simply not part of the dataset.
        data.write_bytes(intact)
        (hub / "dataset/you/first-take/notes.txt").write_text("added on the website")
        fetched = finished(
            receiver, other, share(receiver, other, "fetch", kind="dataset", link="you/first-take")
        )
        saved = receiver.app.state.dataset_library.resolve(fetched["result"]["dataset_id"], other)
        assert hashes(saved) == hashes(recording)

        # A link to nothing, and text that is no link at all.
        missing = finished(
            receiver,
            other,
            share(receiver, other, "fetch", kind="dataset", link="you/nothing"),
            "failed",
        )
        assert "was not found" in missing["message"]
        refused = receiver.post(
            f"/api/v1/projects/{other}/huggingface-sharing/fetch",
            json={"kind": "dataset", "link": "https://example.com/you/first-take"},
        )
        assert refused.status_code == 422 and "Paste" in refused.json()["detail"]


def test_publishing_is_private_unless_public_is_chosen_and_never_touches_other_work(hub, tmp_path):
    recording = tmp_path / "recording"
    build_dataset(recording, episodes=2)
    with machine(tmp_path, "sender") as sender:
        pid = project(sender)
        original = save_dataset(sender, pid, recording, "Second take")

        # A repository of that name already holds something else: it is left alone.
        other_work = hub / "dataset/you/second-take"
        other_work.mkdir(parents=True)
        (other_work / "thesis.pdf").write_bytes(b"not a dataset")
        (hub / "dataset/you/second-take.settings.json").write_text('{"private": true}')
        refused = finished(
            sender, pid, share(sender, pid, "publish", kind="dataset", id=original["id"]), "failed"
        )
        assert "was not published from OpenJensen" in refused["message"]
        assert sorted(path.name for path in other_work.iterdir()) == ["thesis.pdf"]

        # A public repository never receives files that were meant to be private.
        public = hub / "dataset/you/open-take"
        public.mkdir(parents=True)
        (hub / "dataset/you/open-take.settings.json").write_text('{"private": false}')
        kept = finished(
            sender,
            pid,
            share(
                sender, pid, "publish", kind="dataset", id=original["id"], repo_id="you/open-take"
            ),
            "failed",
        )
        assert "already public" in kept["message"] and list(public.iterdir()) == []

        # Public is a separate, explicit choice.
        chosen = finished(
            sender,
            pid,
            share(
                sender,
                pid,
                "publish",
                kind="dataset",
                id=original["id"],
                repo_id="you/open-take",
                private=False,
            ),
        )
        assert chosen["result"]["private"] is False and chosen["inputs"]["private"] is False
        assert (public / "meta/info.json").is_file()

        # Someone else's account: the token may not write there.
        denied = finished(
            sender,
            pid,
            share(
                sender,
                pid,
                "publish",
                kind="dataset",
                id=original["id"],
                repo_id="someone-else/second-take",
            ),
            "failed",
        )
        assert "cannot write here" in denied["message"]
        invalid = sender.post(
            f"/api/v1/projects/{pid}/huggingface-sharing/publish",
            json={"kind": "dataset", "id": original["id"], "repo_id": "not a name"},
        )
        assert invalid.status_code == 422


def test_without_a_saved_token_the_login_of_this_computer_is_used(hub, tmp_path, monkeypatch):
    recording = tmp_path / "recording"
    build_dataset(recording, episodes=2)
    login = tmp_path / "hf-login" / "token"
    monkeypatch.setenv("HF_TOKEN_PATH", str(login))
    with machine(tmp_path, "sender", token=None) as sender:
        pid = project(sender)
        original = save_dataset(sender, pid, recording, "Third take")
        where = f"/api/v1/projects/{pid}/huggingface-sharing/destination"
        query = {"kind": "dataset", "id": original["id"]}

        # Nobody is signed in anywhere: the page can say how to connect, and nothing starts.
        nobody = sender.get(where, params=query).json()
        assert nobody["account"]["connected"] is False and nobody["repo_id"] is None
        assert "save a token in Settings" in nobody["account"]["message"]
        refused = sender.post(
            f"/api/v1/projects/{pid}/huggingface-sharing/publish",
            json={"kind": "dataset", "id": original["id"]},
        )
        assert refused.status_code == 409 and "hf auth login" in refused.json()["detail"]
        assert sender.get(f"/api/v1/projects/{pid}/work-jobs").json()[0]["operation"] != (
            "dataset.publish"
        )

        # ``hf auth login`` was run on this computer: its account is used, and the
        # application never reads or passes the token itself.
        login.parent.mkdir()
        login.write_text("hf_" + "m" * 30)
        signed_in = sender.get(where, params=query).json()
        assert signed_in["account"]["source"] == "machine"
        assert signed_in["repo_id"] == "machine-login/third-take"
        assert sender.get("/api/v1/huggingface-sharing/account").json()["username"] == (
            "machine-login"
        )
        done = finished(
            sender, pid, share(sender, pid, "publish", kind="dataset", id=original["id"])
        )
        assert done["result"]["url"].endswith("/datasets/machine-login/third-take")
        assert [call["token"] for call in calls(hub)] == [None, None]
        assert nowhere_but_the_credential(tmp_path / "sender", "hf_" + "m" * 30) == []


def test_a_token_that_cannot_write_fails_the_job_with_the_reason(hub, tmp_path):
    recording = tmp_path / "recording"
    build_dataset(recording, episodes=2)
    with machine(tmp_path, "sender", token="hf_readonly" + "r" * 20) as sender:
        pid = project(sender)
        original = save_dataset(sender, pid, recording, "Fourth take")
        failed = finished(
            sender, pid, share(sender, pid, "publish", kind="dataset", id=original["id"]), "failed"
        )
        assert "cannot write here" in failed["message"]
        assert "Create a token with write access" in failed["message"]
        problem = sender.get(f"/api/v1/projects/{pid}/work-jobs/{failed['id']}/failure").json()
        assert problem["code"] == "HUGGINGFACE_TRANSFER_FAILED"


# --- Models ------------------------------------------------------------------------------
def smolvla_bundle(folder, *, units=False):
    """The files of a trained SmolVLA checkpoint, with the folders a real one carries."""
    metadata = {
        "architecture": "smolvla",
        "training_backend": "lerobot",
        "reload_verified": True,
        "method": "full",
        "step": 8000,
        "camera_keys": [CAMERA],
        "temporal_contract": {"schema_version": 1, "family": "smolvla", "action_fps": 23},
        "dataset": {"source": "local", "fps": 23.0, "total_episodes": 50},
    }
    if units:
        metadata.update(state_units=UNITS, action_units=UNITS, joint_names=JOINTS)
    contents = {
        "checkpoint/manifest.json": b'{"schema_version": 1, "step": 8000}',
        "checkpoint/recipe.json": b'{"model_id": "lerobot/smolvla_base"}',
        "checkpoint/pretrained_model/config.json": b'{"type": "smolvla"}',
        "checkpoint/pretrained_model/model.safetensors": bytes(range(256)) * 512,
        "checkpoint/pretrained_model/assets/smolvlm/config.json": b'{"vision": true}',
        "checkpoint/pretrained_model/assets/smolvlm/tokenizer.json": b'{"vocabulary": []}',
        "checkpoint/pretrained_model/assets/smolvlm/merges.txt": b"a b\n",
        "checkpoint/pretrained_model/tokenizer/tokenizer.json": b'{"vocabulary": [1]}',
        "checkpoint/pretrained_model/tokenizer/chat_template.jinja": b"{{ messages }}",
        "checkpoint/training_state/training_step.json": b'{"step": 8000}',
        "environment.json": b'{"python": "3.14"}',
        "metrics.jsonl": b"",
    }
    for name, content in contents.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(content)
    files = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    manifest = json.dumps({"schema_version": 1, "metadata": metadata, "files": files}).encode()
    (folder / "manifest.json").write_bytes(manifest)
    return metadata, files, hashlib.sha256(manifest).hexdigest(), contents


def trained_model(client, pid, *, units=False, cloud=False):
    """A finished training job in this project whose result is one SmolVLA checkpoint."""
    data = client.app.state.execution.settings.data_dir
    job_id = "11111111-2222-4333-8444-555555555555"
    folder = data / "jobs" / job_id / "operation" / "bundle"
    folder.mkdir(parents=True)
    metadata, files, sha, contents = smolvla_bundle(folder, units=units)
    archive = None
    if cloud:
        # Trained in the cloud: only the manifest and a pointer are on this machine.
        metadata.update(storage="gcs", remote_uri=f"gs://training-bucket/jobs/{job_id}/artifact")
        manifest = json.dumps({"schema_version": 1, "metadata": metadata, "files": files}).encode()
        sha = hashlib.sha256(manifest).hexdigest()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as stream:
            for name, content in {"manifest.json": manifest, **contents}.items():
                item = tarfile.TarInfo("policy/" + name)
                item.size = len(content)
                stream.addfile(item, io.BytesIO(content))
        archive = buffer.getvalue()
        for path in sorted(folder.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        (folder / "manifest.json").write_bytes(manifest)
        (folder / "remote.json").write_text(
            json.dumps(
                {
                    "uri": metadata["remote_uri"],
                    "manifest_sha256": sha,
                    "file_bytes": sum(len(content) for content in contents.values()),
                }
            )
        )
    artifact = PolicyArtifact(
        id=f"{job_id}:operation",
        project_id=pid,
        job_id=job_id,
        label="SmolVLA checkpoint",
        format="training_checkpoint",
        path=folder.relative_to(data).as_posix(),
        manifest_sha256=sha,
        file_bytes=sum(len(content) for content in contents.values()),
        metadata=metadata,
    )
    record = Job(
        id=job_id,
        project_id=pid,
        kind="policy.export",
        status="succeeded",
        request=PolicyRequest(operation="policy.export", runtime_id="native-cpu", artifact_id="x"),
        result=LifecycleResult(artifacts=[artifact]),
        created_at=now(),
        updated_at=now(),
    )

    async def seed():
        async with client.app.state.execution.storage.engine.begin() as connection:
            await connection.execute(
                insert(jobs).values(
                    id=record.id, project_id=pid, status=record.status, record=record.model_dump()
                )
            )

    client.portal.call(seed)
    return artifact, contents, archive


class CloudStream:
    """What cloud storage hands over: the model as one archive, a piece at a time."""

    def __init__(self, archive):
        self.archive, self.closed = archive, False

    async def __aiter__(self):
        for start in range(0, len(self.archive), 7919):
            yield self.archive[start : start + 7919]

    async def aclose(self):
        self.closed = True


def companion(client, received):
    """The robot's USB computer, accepting an installation and keeping what arrived."""

    async def handler(request):
        received["descriptor"] = json.loads(
            base64.b64decode(request.headers["x-openjensen-policy"], validate=True)
        )
        with tarfile.open(fileobj=io.BytesIO(await request.aread())) as archive:
            received["files"] = {
                member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                for member in archive
                if member.isfile()
            }
        return httpx.Response(200, json={"installed_policy_id": received["descriptor"]["id"]})

    data = client.app.state.execution.settings.data_dir
    connection = data / "robot-lab-connection.json"
    connection.write_text(json.dumps({"url": "http://127.0.0.1:18101", "token": COMPANION}))
    connection.chmod(0o600)
    client.app.state.robot_lab = RobotLabBridge(
        data, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


@pytest.mark.parametrize("cloud", [False, True], ids=["saved-here", "saved-in-cloud-storage"])
def test_a_trained_smolvla_checkpoint_installs_on_another_machine_after_a_round_trip(
    hub, tmp_path, monkeypatch, cloud
):
    evidence = {
        "units": "degrees; gripper percent",
        "state_names": [joint + ".pos" for joint in JOINTS],
        "action_names": [joint + ".pos" for joint in JOINTS],
        "dataset_fps": 23,
    }
    with machine(tmp_path, "sender") as sender, machine(tmp_path, "receiver") as receiver:
        pid, other = project(sender), project(receiver, "Run on the arm")
        artifact, contents, archive = trained_model(sender, pid, cloud=cloud)
        sender.put(
            f"/api/v1/projects/{pid}/model-runs/{artifact.job_id}/name", json={"name": "Pen v3"}
        )
        destination = sender.get(
            f"/api/v1/projects/{pid}/huggingface-sharing/destination",
            params={"kind": "model", "id": artifact.id},
        ).json()
        assert destination["repo_id"] == "you/pen-v3" and destination["private"] is True
        assert destination["url"] == "https://huggingface.co/you/pen-v3"

        with monkeypatch.context() as patch:
            # On the training machine the recording itself proves the units and joints.
            async def from_the_recording(app, project_id, model):
                return evidence

            patch.setattr("vla_platform.robot_policy_install.recorded_control", from_the_recording)
            stream = CloudStream(archive) if cloud else None
            if cloud:

                async def download(project_id, artifact_id):
                    return stream

                patch.setattr(sender.app.state.execution.lifecycle, "download", download)
            published = finished(
                sender, pid, share(sender, pid, "publish", kind="model", id=artifact.id)
            )
            assert stream is None or stream.closed
        assert published["operation"] == "policy.publish" and published["module"] == "models"
        assert [event["stage"] for event in published["events"]] == [
            *(["collecting"] if cloud else []),
            "preparing",
            "uploading",
            "succeeded",
        ]
        assert published["result"]["private"] is True
        assert published["result"]["manifest_sha256"] == artifact.manifest_sha256
        repository = hub / "model/you/pen-v3"
        sent = hashes(repository)
        for name, content in contents.items():
            assert sent[name] == hashlib.sha256(content).hexdigest()
        assert sent["manifest.json"] == artifact.manifest_sha256
        assert not (tmp_path / "sender/huggingface-work").exists()

        fetched = finished(
            receiver,
            other,
            share(receiver, other, "fetch", kind="model", link=published["result"]["url"]),
        )
        assert fetched["operation"] == "policy.fetch"
        (model,) = receiver.get(f"/api/v1/projects/{other}/artifacts").json()
        assert model["id"] == fetched["result"]["artifact_id"]
        assert model["manifest_sha256"] == artifact.manifest_sha256
        assert model["metadata"] == artifact.metadata and model["format"] == "training_checkpoint"
        assert model["label"] == "SmolVLA checkpoint" and model["run_name"] == "Pen v3"
        assert model["file_bytes"] == artifact.file_bytes and model["project_id"] == other
        saved = tmp_path / "receiver" / model["path"]
        assert hashes(saved) == {
            **{name: hashlib.sha256(content).hexdigest() for name, content in contents.items()},
            "manifest.json": artifact.manifest_sha256,
        }
        assert (saved / "checkpoint/pretrained_model/assets/smolvlm/config.json").is_file()
        assert (saved / "checkpoint/pretrained_model/tokenizer/tokenizer.json").is_file()

        # It can be renamed, and the same link added again is the same model.
        renamed = receiver.put(
            f"/api/v1/projects/{other}/model-runs/{model['job_id']}/name", json={"name": "Pen"}
        )
        assert renamed.status_code == 200, renamed.text
        twice = finished(
            receiver, other, share(receiver, other, "fetch", kind="model", link="you/pen-v3")
        )
        assert twice["result"]["artifact_id"] == model["id"]
        assert twice["result"]["already_saved"] is True
        assert len(receiver.get(f"/api/v1/projects/{other}/artifacts").json()) == 1

        # Run policy: the model installs on the arm's computer, byte for byte, and with
        # the units the training recording proved, although that recording is not here.
        received = {}
        companion(receiver, received)
        installed = receiver.post(
            f"/api/v1/projects/{other}/robot-lab/policies/{model['id']}/install",
            json={"name": "Pen policy"},
        )
        assert installed.status_code == 201, installed.text
        descriptor = received["descriptor"]
        assert descriptor["policy_type"] == "smolvla" and descriptor["offline_only"] is False
        assert descriptor["fps"] == 23 and descriptor["state_units"] == UNITS
        assert descriptor["source_manifest_sha256"] == artifact.manifest_sha256
        assert received["files"] == {"policy/" + name: sha for name, sha in hashes(saved).items()}
        assert (
            receiver.portal.call(
                recorded_control,
                receiver.app,
                other,
                PolicyArtifact.model_validate({**model, "run_name": None}),
            )
            == evidence
        )
        receiver.portal.call(receiver.app.state.robot_lab.close)

    for name in ("sender", "receiver"):
        assert nowhere_but_the_credential(tmp_path / name, TOKEN) == []


def test_a_model_that_changed_on_the_hub_or_was_published_elsewhere_is_not_added(hub, tmp_path):
    with machine(tmp_path, "sender") as sender, machine(tmp_path, "receiver") as receiver:
        pid, other = project(sender), project(receiver)
        artifact, _, _ = trained_model(sender, pid, units=True)
        finished(sender, pid, share(sender, pid, "publish", kind="model", id=artifact.id))
        weights = hub / "model/you/smolvla-8000-sft/checkpoint/pretrained_model/model.safetensors"
        weights.write_bytes(weights.read_bytes()[:-1] + b"!")
        changed = finished(
            receiver,
            other,
            share(receiver, other, "fetch", kind="model", link="you/smolvla-8000-sft"),
            "failed",
        )
        assert "differs from what was published" in changed["message"]
        assert "model.safetensors" in changed["message"]

        # An ordinary model repository from the Hub is not an OpenJensen bundle.
        ordinary = hub / "model/lerobot/smolvla_base"
        ordinary.mkdir(parents=True)
        (ordinary / "config.json").write_text("{}")
        foreign = finished(
            receiver,
            other,
            share(receiver, other, "fetch", kind="model", link="lerobot/smolvla_base"),
            "failed",
        )
        assert "was not published from OpenJensen" in foreign["message"]
        assert receiver.get(f"/api/v1/projects/{other}/artifacts").json() == []
        assert not (tmp_path / "receiver/jobs").exists()
        missing = sender.post(
            f"/api/v1/projects/{pid}/huggingface-sharing/publish",
            json={"kind": "model", "id": "no-such-model"},
        )
        assert missing.status_code == 404


def test_a_slow_upload_says_it_is_alive_and_a_stalled_cloud_fetch_gives_up_with_a_reason(
    hub, tmp_path, monkeypatch
):
    class Stalled(CloudStream):
        async def __aiter__(self):
            yield self.archive[:4096]
            await asyncio.sleep(30)

    monkeypatch.setattr(sharing, "HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setenv("FAKE_HUB_UPLOAD_DELAY", "1")
    with machine(tmp_path, "sender") as sender:
        pid = project(sender)
        artifact, _, archive = trained_model(sender, pid, units=True, cloud=True)
        lifecycle = sender.app.state.execution.lifecycle

        # The connection is slow, not broken: the job keeps reporting while it uploads.
        async def download(project_id, artifact_id):
            return CloudStream(archive)

        monkeypatch.setattr(lifecycle, "download", download)
        started = share(sender, pid, "publish", kind="model", id=artifact.id)
        seen = set()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            value = sender.get(f"/api/v1/projects/{pid}/work-jobs/{started['id']}").json()
            seen.add(value["message"])
            if value["status"] != "running":
                break
            time.sleep(0.02)
        assert value["status"] == "succeeded", value
        assert any(message and message.endswith("min so far") for message in seen), seen

        # Cloud storage stops answering part way: the job ends with what to do next.
        stalled = Stalled(archive)

        async def stall(project_id, artifact_id):
            return stalled

        monkeypatch.setattr(lifecycle, "download", stall)
        monkeypatch.setattr("vla_platform.robot_policy_install.DOWNLOAD_SECONDS", 0.3)
        failed = finished(
            sender,
            pid,
            share(sender, pid, "publish", kind="model", id=artifact.id, repo_id="you/second"),
            "failed",
        )
        assert "could not be fetched from storage" in failed["message"]
        assert "publish it again" in failed["message"] and stalled.closed
        assert not (hub / "model/you/second").exists()
        assert not (tmp_path / "sender/huggingface-work").exists()


def test_a_transfer_can_be_cancelled_and_one_cut_off_by_a_restart_says_so(
    hub, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_HUB_DELAY", "30")
    (hub / "dataset/you/slow").mkdir(parents=True)
    with machine(tmp_path, "receiver") as receiver:
        pid = project(receiver)
        started = share(receiver, pid, "fetch", kind="dataset", link="you/slow")
        assert started["status"] == "running" and started["can_cancel"] is True
        # A second click joins the transfer that is already under way.
        assert share(receiver, pid, "fetch", kind="dataset", link="you/slow")["id"] == started["id"]
        cancelled = receiver.post(f"/api/v1/projects/{pid}/work-jobs/{started['id']}/cancel")
        assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
        assert (
            receiver.post(f"/api/v1/projects/{pid}/work-jobs/{started['id']}/cancel").status_code
            == 409
        )
        left_running = share(receiver, pid, "fetch", kind="dataset", link="you/slow")
    # The application was closed while that one was still running.
    journal = sharing.ShareJobs(tmp_path / "receiver")
    stopped = journal.get(left_running["source_id"])
    assert stopped["status"] == "interrupted" and "Start it again" in stopped["error"]
    # A crash leaves no chance to say so: the next start finds the record and closes it.
    crashed = journal.create(pid, "policy.fetch", "you/pen", {"repo_id": "you/pen"})
    (tmp_path / "receiver/huggingface-work" / crashed["id"]).mkdir(parents=True)
    with TestClient(create_app(Settings(data_dir=tmp_path / "receiver"))) as restarted:
        found = restarted.get(f"/api/v1/projects/{pid}/work-jobs/share:{crashed['id']}").json()
        assert found["status"] == "interrupted" and found["can_cancel"] is False
        assert "Start it again" in found["message"]
    assert not (tmp_path / "receiver/huggingface-work").exists()


# --- The client process itself -------------------------------------------------------------
class Repository:
    def __init__(self, files, private):
        self.siblings = [type("File", (), {"rfilename": name})() for name in files]
        self.private, self.sha = private, "c" * 40


class Hub:
    """The few calls the worker makes, recorded in order."""

    def __init__(self, existing=None, private=True, lose=()):
        self.files = None if existing is None else set(existing)
        self.private, self.lose, self.log = private, set(lose), []

    def repo_info(self, repo, repo_type):
        if self.files is None:
            raise type("RepositoryNotFoundError", (Exception,), {})()
        return Repository(self.files, self.private)

    def create_repo(self, repo, repo_type, private, exist_ok):
        self.files, self.private = {".gitattributes"}, private
        self.log.append(("create", private))

    def update_repo_settings(self, repo, private, repo_type):
        self.private = private
        self.log.append(("visibility", private))

    def upload_file(self, path_or_fileobj, path_in_repo, **_):
        self.files.add(path_in_repo)
        self.log.append(("file", path_in_repo, json.loads(path_or_fileobj).get("state")))

    def upload_folder(self, folder_path, ignore_patterns, **_):
        self.files |= {"data.bin", "meta/info.json"} - self.lose
        self.log.append(("folder", folder_path))


def push(hub, **changes):
    return worker.push(
        hub,
        {
            "repo_type": "dataset",
            "repo_id": "you/pick-pen",
            "folder": "/folder",
            "private": True,
            "message": "Publish",
            "begin": {sharing.MARKER: '{"state": "publishing"}'},
            "files": {sharing.MARKER: '{"state": "published"}'},
            "marker": sharing.MARKER,
            "expect": ["data.bin", "meta/info.json"],
            **changes,
        },
    )


def test_the_client_marks_a_repository_before_and_after_and_respects_what_exists(capsys):
    fresh = Hub()
    assert push(fresh) == {"repo_id": "you/pick-pen", "revision": "c" * 40, "private": True}
    assert fresh.log == [
        ("create", True),
        ("file", sharing.MARKER, "publishing"),
        ("folder", "/folder"),
        ("file", sharing.MARKER, "published"),
    ]
    assert '{"progress": {"stage": "uploading"}}' in capsys.readouterr().out

    someone_elses = Hub(existing={".gitattributes", "thesis.pdf"})
    assert "was not published from OpenJensen" in push(someone_elses)["error"]
    already_public = Hub(existing={sharing.MARKER, "data.bin"}, private=False)
    assert "already public" in push(already_public)["error"]
    assert someone_elses.log == [] and already_public.log == []

    # An interrupted publish left only the first file: publishing again is allowed.
    resumed = Hub(existing={".gitattributes", sharing.MARKER})
    assert "error" not in push(resumed) and resumed.log[0][0] == "file"
    # Making a private repository public happens only on that explicit choice.
    opened = Hub(existing={sharing.MARKER})
    assert push(opened, private=False)["private"] is False
    assert opened.log[0] == ("visibility", False)
    # A file the Hub did not keep is a failed publish, not a quiet gap.
    lossy = Hub(lose={"meta/info.json"})
    assert "did not keep 1 of the files, for example meta/info.json" in push(lossy)["error"]


def test_the_client_explains_refusals_without_ever_naming_a_token():
    def refusal(status):
        error = Exception("hf_secretsecretsecret")
        error.response = type("Response", (), {"status_code": status})()
        return worker.explain(error)

    assert refusal(403)["status"] == 403 and "write access" in refusal(403)["error"]
    assert "rejected the saved token" in refusal(401)["error"]
    assert "was not found" in refusal(404)["error"]
    missing = worker.explain(type("LocalTokenNotFoundError", (Exception,), {})())
    assert "hf auth login" in missing["error"]
    assert all("hf_secret" not in json.dumps(refusal(status)) for status in (401, 403, 404, 500))


def test_an_inventory_is_the_identity_the_dataset_reader_gives_the_same_folder(tmp_path):
    from test_dataset_library import run_worker

    folder = tmp_path / "recording"
    build_dataset(folder, episodes=2)
    (folder / "a-b").mkdir()
    (folder / "a-b/late.txt").write_text("sorts after a/ by path, before it by text")
    (folder / "a").mkdir()
    (folder / "a/early.txt").write_text("x")
    identity, files = sharing.inventory(folder)
    assert identity == run_worker(folder, "detect")["sha256"]
    assert [row[0] for row in files][:2] == ["a/early.txt", "a-b/late.txt"]


def test_sizes_are_written_the_way_a_person_reads_them():
    assert sharing.size_text(300) == "1 KB" and sharing.size_text(133_000) == "130 KB"
    assert sharing.size_text(483 * 1024**2) == "483.0 MB"
    assert sharing.size_text(1_329_790_998) == "1.24 GB"
