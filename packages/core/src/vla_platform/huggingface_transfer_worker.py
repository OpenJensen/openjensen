"""Move one folder to or from the Hugging Face Hub with the official client.

Runs as its own process in an environment that has ``huggingface_hub``; the application
never imports it. The request is one JSON object on standard input, the result one JSON
object on standard output, after any ``{"progress": ...}`` lines. The access token
arrives only in ``HF_TOKEN``: never on the command line, never in the request, never in
the result. Without one, the client uses the login ``hf auth login`` saved on this
computer.
"""

import json
import os
import sys
from pathlib import Path

# Local bookkeeping the client writes beside downloads; never part of a dataset or model.
IGNORED = [".cache/**", "**/.DS_Store", ".git/**", "**/__pycache__/**"]


def explain(error) -> dict:
    status = getattr(getattr(error, "response", None), "status_code", None)
    name = type(error).__name__
    if name == "LocalTokenNotFoundError":
        return {
            "status": 401,
            "error": "No Hugging Face account is connected. Save a token in Settings, "
            "or run hf auth login on this computer.",
        }
    if status == 401:
        return {
            "status": 401,
            "error": "Hugging Face rejected the saved token. Save a valid token in Settings.",
        }
    if status == 403:
        return {
            "status": 403,
            "error": "The saved Hugging Face token cannot write here. "
            "Create a token with write access and save it in Settings.",
        }
    if status == 404 or name in {"RepositoryNotFoundError", "RevisionNotFoundError"}:
        return {
            "status": 404,
            "error": "That Hugging Face repository was not found, "
            "or the saved token cannot read it.",
        }
    if name in {
        "ConnectionError",
        "ConnectError",
        "Timeout",
        "ReadTimeout",
        "OfflineModeIsEnabled",
    }:
        return {"error": "Could not reach Hugging Face. Check the connection and try again."}
    return {"error": f"Hugging Face transfer failed ({name})."}


def report(**progress):
    print(json.dumps({"progress": progress}), flush=True)


def names(info):
    return {item.rfilename for item in info.siblings or []}


def whoami(api, request):
    return {"username": api.whoami()["name"]}


def push(api, request):
    kind, repo, folder = request["repo_type"], request["repo_id"], Path(request["folder"])
    private, marker = bool(request["private"]), request.get("marker")
    try:
        existing = api.repo_info(repo, repo_type=kind)
    except Exception as error:
        if type(error).__name__ != "RepositoryNotFoundError":
            raise
        existing = None
    if existing is None:
        api.create_repo(repo, repo_type=kind, private=private, exist_ok=True)
    else:
        # Never write into a repository that holds someone's other work, and never
        # put files meant to be private into one that is already public.
        held = names(existing) - {".gitattributes"}
        if marker and held and marker not in held:
            return {
                "status": 409,
                "error": f"{repo} already exists on Hugging Face and was not published "
                "from OpenJensen. Choose another name.",
            }
        if private and not existing.private:
            return {
                "status": 409,
                "error": f"{repo} is already public on Hugging Face. Choose another name to "
                "publish privately, or make that repository private there first.",
            }
        if not private and existing.private:
            api.update_repo_settings(repo, private=False, repo_type=kind)

    def upload(files, stage):
        for path, text in sorted(files.items()):
            api.upload_file(
                path_or_fileobj=text.encode(),
                path_in_repo=path,
                repo_id=repo,
                repo_type=kind,
                commit_message=f"{stage} {path}",
            )

    upload(request.get("begin") or {}, "Begin")
    report(stage="uploading")
    api.upload_folder(
        repo_id=repo,
        repo_type=kind,
        folder_path=str(folder),
        commit_message=request["message"],
        ignore_patterns=list(IGNORED),
    )
    upload(request.get("files") or {}, "Add")
    info = api.repo_info(repo, repo_type=kind)
    missing = sorted(set(request.get("expect") or []) - names(info))
    if missing:
        return {
            "error": f"Hugging Face did not keep {len(missing)} of the files, for example "
            f"{missing[0]}. The repository is incomplete; publish again."
        }
    if tag := request.get("tag"):
        # Loaders find a dataset by its format tag, so the tag follows the newest upload.
        try:
            api.delete_tag(repo, tag=tag, repo_type=kind)
        except Exception:
            pass
        api.create_tag(repo, tag=tag, revision=info.sha, repo_type=kind)
    return {"repo_id": repo, "revision": info.sha, "private": bool(info.private)}


def pull(api, request):
    from huggingface_hub import snapshot_download

    kind, repo = request["repo_type"], request["repo_id"]
    revision = request.get("revision") or None
    info = api.repo_info(repo, repo_type=kind, revision=revision, files_metadata=True)
    files = [item for item in info.siblings or [] if not item.rfilename.startswith(".git")]
    total = sum(item.size or 0 for item in files)
    if len(files) > request["max_files"] or total > request["max_bytes"]:
        return {
            "error": f"This repository holds {len(files)} files and {total / 1024**3:.1f} GiB; "
            f"an import accepts at most {request['max_files']} files and "
            f"{request['max_bytes'] / 1024**3:.0f} GiB."
        }
    report(stage="downloading", files=len(files), bytes=total)
    snapshot_download(
        repo,
        repo_type=kind,
        revision=info.sha,
        local_dir=request["folder"],
        ignore_patterns=[".gitattributes"],
    )
    return {"repo_id": repo, "revision": info.sha, "private": bool(info.private)}


def main():
    try:
        request = json.loads(sys.stdin.buffer.read(4 * 1024**2))
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        operation = {"push": push, "pull": pull, "whoami": whoami}[request["operation"]]
        result = operation(api, request)
    except Exception as error:
        result = explain(error)
    print(json.dumps(result), flush=True)
    return 1 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
