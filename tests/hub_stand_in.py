"""A stand-in for the Hugging Face client process: "the Hub" is a local folder.

Tests run this file where the application would run ``huggingface_transfer_worker.py``.
It speaks the same protocol (one JSON request on standard input, progress lines and one
JSON answer on standard output, the token only in ``HF_TOKEN``) and keeps the rules of
the real Hub that the application depends on:

- a repository is private or public, and a private one needs an account to read;
- writing needs an account that may write;
- a new repository starts with a ``.gitattributes`` the publisher never sent;
- a download leaves the client's own bookkeeping in a ``.cache`` folder;
- files the client never uploads (``.cache``, ``.DS_Store``) do not arrive.

``FAKE_HUB`` names the folder. Tests reach into it to look at what was sent, or to
damage a repository before it is fetched. Every call is appended to ``calls.jsonl``.

Knobs for failure cases: a token containing ``readonly`` may not write and one
containing ``invalid`` is rejected; ``FAKE_HUB_STOP_AFTER_BEGIN`` names a repository
whose upload dies after its first file; ``FAKE_HUB_DELAY`` slows every transfer before
it starts and ``FAKE_HUB_UPLOAD_DELAY`` slows an upload once it is under way.
"""

import fnmatch
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

NEVER_UPLOADED = [".cache/*", "*/.DS_Store", ".DS_Store", ".git/*", "*/__pycache__/*"]


def answer(value):
    print(json.dumps(value), flush=True)
    sys.exit(1 if "error" in value else 0)


def account(token):
    """Who is asking: the token's account, this computer's saved login, or nobody."""
    if token:
        if "invalid" in token:
            answer(
                {
                    "status": 401,
                    "error": "Hugging Face rejected the saved token. "
                    "Save a valid token in Settings.",
                }
            )
        return os.environ.get("FAKE_HUB_USER", "you")
    login = Path(os.environ.get("HF_TOKEN_PATH", "/nonexistent"))
    if login.is_file() and login.stat().st_size:
        return os.environ.get("FAKE_HUB_MACHINE_USER", "machine-login")
    return None


def files_of(folder):
    return {
        path.relative_to(folder).as_posix(): path
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def revision(folder):
    listing = [
        [name, hashlib.sha256(path.read_bytes()).hexdigest()]
        for name, path in files_of(folder).items()
    ]
    return hashlib.sha1(json.dumps(listing).encode()).hexdigest()


def main():
    hub = Path(os.environ["FAKE_HUB"])
    request = json.loads(sys.stdin.read())
    token = os.environ.get("HF_TOKEN")
    with (hub / "calls.jsonl").open("a") as log:
        log.write(json.dumps({"argv": sys.argv, "request": request, "token": token}) + "\n")
    user = account(token)
    if request["operation"] == "whoami":
        if user is None:
            answer(
                {
                    "status": 401,
                    "error": "No Hugging Face account is connected. Save a token in Settings, "
                    "or run hf auth login on this computer.",
                }
            )
        answer({"username": user})
    time.sleep(float(os.environ.get("FAKE_HUB_DELAY", "0")))
    repo = request["repo_id"]
    folder = hub / request["repo_type"] / repo
    settings = hub / request["repo_type"] / (repo + ".settings.json")
    if request["operation"] == "push":
        if user is None:
            answer({"status": 401, "error": "No Hugging Face account is connected."})
        if "readonly" in (token or "") or repo.split("/")[0] == "someone-else":
            answer(
                {
                    "status": 403,
                    "error": "The saved Hugging Face token cannot write here. "
                    "Create a token with write access and save it in Settings.",
                }
            )
        private, marker = bool(request["private"]), request.get("marker")
        if folder.is_dir():
            saved = json.loads(settings.read_text())
            held = set(files_of(folder)) - {".gitattributes"}
            if marker and held and marker not in held:
                answer(
                    {
                        "status": 409,
                        "error": f"{repo} already exists on Hugging Face and was not published "
                        "from OpenJensen. Choose another name.",
                    }
                )
            if private and not saved["private"]:
                answer(
                    {
                        "status": 409,
                        "error": f"{repo} is already public on Hugging Face. Choose another "
                        "name to publish privately, or make that repository private there first.",
                    }
                )
        else:
            folder.mkdir(parents=True)
            (folder / ".gitattributes").write_text("*.safetensors filter=lfs\n")
        settings.write_text(json.dumps({"private": private}))

        def write(files):
            for name, text in files.items():
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_text(text)

        write(request.get("begin") or {})
        if os.environ.get("FAKE_HUB_STOP_AFTER_BEGIN") == repo:
            answer({"error": "Could not reach Hugging Face. Check the connection and try again."})
        print(json.dumps({"progress": {"stage": "uploading"}}), flush=True)
        time.sleep(float(os.environ.get("FAKE_HUB_UPLOAD_DELAY", "0")))
        for name, path in files_of(Path(request["folder"])).items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in NEVER_UPLOADED):
                continue
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, folder / name)
        write(request.get("files") or {})
        missing = sorted(set(request.get("expect") or []) - set(files_of(folder)))
        if missing:
            answer(
                {
                    "error": f"Hugging Face did not keep {len(missing)} of the files, for "
                    f"example {missing[0]}. The repository is incomplete; publish again."
                }
            )
        if request.get("tag"):
            (hub / request["repo_type"] / (repo + ".tag")).write_text(request["tag"])
        answer({"repo_id": repo, "revision": revision(folder), "private": private})
    private = json.loads(settings.read_text())["private"] if settings.is_file() else False
    if not folder.is_dir() or (private and user is None):
        answer(
            {
                "status": 404,
                "error": "That Hugging Face repository was not found, "
                "or the saved token cannot read it.",
            }
        )
    held = {name: path for name, path in files_of(folder).items() if not name.startswith(".git")}
    total = sum(path.stat().st_size for path in held.values())
    if len(held) > request["max_files"] or total > request["max_bytes"]:
        answer(
            {
                "error": f"This repository holds {len(held)} files and {total / 1024**3:.1f} GiB; "
                f"an import accepts at most {request['max_files']} files and "
                f"{request['max_bytes'] / 1024**3:.0f} GiB."
            }
        )
    print(
        json.dumps({"progress": {"stage": "downloading", "files": len(held), "bytes": total}}),
        flush=True,
    )
    destination = Path(request["folder"])
    for name, path in held.items():
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination / name)
    bookkeeping = destination / ".cache/huggingface/download"
    bookkeeping.mkdir(parents=True, exist_ok=True)
    (bookkeeping / "files.metadata").write_text("the client's own notes\n")
    answer({"repo_id": repo, "revision": revision(folder), "private": private})


if __name__ == "__main__":
    main()
