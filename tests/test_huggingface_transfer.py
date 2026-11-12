"""Moving a dataset or model through Hugging Face, without ever touching Hugging Face.

The real client runs in its own process. Here that process is a small stand-in that
keeps "the Hub" in a local folder, so the contract is exercised end to end: what is
sent, what comes back, and where the access token is allowed to appear.
"""

import asyncio
import json
import sys

import pytest
from vla_platform import huggingface_transfer as transfer
from vla_platform.huggingface_transfer import TransferError, parse_reference, slug

TOKEN = "hf_" + "t" * 30
FAKE_CLIENT = r"""
import json, os, shutil, sys
from pathlib import Path

hub = Path(os.environ["FAKE_HUB"])
request = json.loads(sys.stdin.read())
(hub / "calls.jsonl").open("a").write(
    json.dumps({"argv": sys.argv, "request": request, "token": os.environ.get("HF_TOKEN")}) + "\n"
)
repo = hub / request["repo_type"] / request["repo_id"]
if request["repo_id"].endswith("/denied"):
    print(json.dumps({"error": "The saved Hugging Face token cannot write here."}))
    sys.exit(1)
if request["repo_id"].endswith("/leaky"):
    print(json.dumps({"error": "boom " + os.environ["HF_TOKEN"]}))
    sys.exit(1)
if request["repo_id"].endswith("/silent"):
    sys.exit(0)
if request["operation"] == "push":
    shutil.copytree(request["folder"], repo, dirs_exist_ok=True)
    for name, text in request["files"].items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
else:
    if not repo.is_dir():
        print(json.dumps({"error": "That Hugging Face repository was not found."}))
        sys.exit(1)
    shutil.copytree(repo, request["folder"], dirs_exist_ok=True)
print(json.dumps({"repo_id": request["repo_id"], "revision": "a" * 40, "private": True}))
"""


@pytest.fixture
def hub(tmp_path, monkeypatch):
    client = tmp_path / "fake_client.py"
    client.write_text(FAKE_CLIENT)
    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setenv("FAKE_HUB", str(root))
    monkeypatch.setenv("JENSEN_HUGGINGFACE_PYTHON", sys.executable)
    monkeypatch.setattr(transfer, "WORKER", client)
    return root


def calls(hub):
    return [json.loads(line) for line in (hub / "calls.jsonl").read_text().splitlines()]


@pytest.mark.parametrize(
    "text,kind,expected",
    [
        ("you/pick-pen", "dataset", ("you/pick-pen", None)),
        ("  https://huggingface.co/datasets/you/pick-pen  ", "dataset", ("you/pick-pen", None)),
        ("https://huggingface.co/datasets/you/pick-pen/", "dataset", ("you/pick-pen", None)),
        ("hf.co/datasets/you/pick-pen", "dataset", ("you/pick-pen", None)),
        ("datasets/you/pick-pen", "dataset", ("you/pick-pen", None)),
        (
            "https://huggingface.co/datasets/you/pick-pen/tree/v3.0",
            "dataset",
            ("you/pick-pen", "v3.0"),
        ),
        ("https://huggingface.co/you/act-pen?library=lerobot", "model", ("you/act-pen", None)),
        (
            "https://huggingface.co/you/act-pen/blob/main/config.json",
            "model",
            ("you/act-pen", "main"),
        ),
    ],
)
def test_people_paste_links_in_many_shapes(text, kind, expected):
    assert parse_reference(text, kind) == expected


@pytest.mark.parametrize(
    "text,kind",
    [
        ("", "dataset"),
        ("pick-pen", "dataset"),
        ("https://example.com/you/pick-pen", "dataset"),
        ("you/../secrets", "dataset"),
        ("https://huggingface.co/spaces/you/demo", "model"),
        ("https://huggingface.co/datasets/you/pick-pen", "model"),
        ("you/pick pen", "dataset"),
        ("you/pick-pen/extra/path", "dataset"),
    ],
)
def test_unusable_links_are_refused_with_a_reason(text, kind):
    with pytest.raises(TransferError):
        parse_reference(text, kind)


def test_dataset_names_become_readable_repository_names():
    assert slug("Pick up the pen!") == "pick-up-the-pen"
    assert slug("  First take (2) ") == "first-take-2"
    assert slug("!!!") == "untitled"
    assert len(slug("x" * 300)) == 96


def test_publish_then_fetch_on_another_machine_returns_the_same_files(hub, tmp_path):
    source = tmp_path / "dataset"
    (source / "meta").mkdir(parents=True)
    (source / "meta/info.json").write_text('{"codebase_version": "v3.0"}')
    (source / "data.bin").write_bytes(bytes(range(256)) * 64)

    async def round_trip():
        published = await transfer.push(
            "dataset",
            source,
            "you/pick-pen",
            TOKEN,
            files={"README.md": "# Pick pen\n"},
            tag="v3.0",
        )
        fetched = await transfer.pull("dataset", "you/pick-pen", tmp_path / "copy", TOKEN)
        return published, fetched

    published, fetched = asyncio.run(round_trip())
    assert published == {
        "repo_id": "you/pick-pen",
        "revision": "a" * 40,
        "private": True,
        "url": "https://huggingface.co/datasets/you/pick-pen",
    }
    assert fetched["revision"] == published["revision"]
    copy = tmp_path / "copy"
    assert (copy / "data.bin").read_bytes() == (source / "data.bin").read_bytes()
    assert (copy / "README.md").read_text() == "# Pick pen\n"
    push, pull = calls(hub)
    assert push["request"]["private"] is True and push["request"]["tag"] == "v3.0"
    assert pull["request"]["max_bytes"] == 2 * 1024**3


def test_the_token_reaches_the_client_only_through_its_environment(hub, tmp_path):
    (tmp_path / "model").mkdir()
    asyncio.run(transfer.push("model", tmp_path / "model", "you/act-pen", TOKEN))
    (call,) = calls(hub)
    assert call["token"] == TOKEN
    assert TOKEN not in json.dumps(call["argv"]) and TOKEN not in json.dumps(call["request"])
    assert transfer.repository_url("model", "you/act-pen") == "https://huggingface.co/you/act-pen"


def test_a_refusal_from_hugging_face_reaches_the_person_without_the_token(hub, tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(TransferError, match="cannot write here"):
        asyncio.run(transfer.push("dataset", tmp_path / "data", "you/denied", TOKEN))
    with pytest.raises(TransferError) as leaked:
        asyncio.run(transfer.push("dataset", tmp_path / "data", "you/leaky", TOKEN))
    assert TOKEN not in str(leaked.value)
    with pytest.raises(TransferError, match="returned no result"):
        asyncio.run(transfer.push("dataset", tmp_path / "data", "you/silent", TOKEN))
    with pytest.raises(TransferError, match="not found"):
        asyncio.run(transfer.pull("dataset", "you/missing", tmp_path / "none", None))


def test_publishing_needs_a_connected_account_and_a_valid_name(hub, tmp_path):
    with pytest.raises(TransferError, match="Connect a Hugging Face account") as missing:
        asyncio.run(transfer.push("dataset", tmp_path, "you/pick-pen", ""))
    assert missing.value.status == 409
    with pytest.raises(TransferError, match="repository name"):
        asyncio.run(transfer.push("dataset", tmp_path, "not a name", TOKEN))
    assert not (hub / "calls.jsonl").exists()
