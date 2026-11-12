"""Move one folder to or from the Hugging Face Hub with the official client.

Runs as its own process in an environment that has ``huggingface_hub``; the application
never imports it. The request is one JSON object on standard input, the result one JSON
object on standard output. The access token arrives only in ``HF_TOKEN``: never on the
command line, never in the request, never in the result.
"""

import json
import os
import sys
from pathlib import Path

# Local bookkeeping the client writes beside downloads; never part of a dataset or model.
IGNORED = [".cache/**", "**/.DS_Store", ".git/**", "**/__pycache__/**"]


def explain(error) -> str:
    status = getattr(getattr(error, "response", None), "status_code", None)
    name = type(error).__name__
    if status == 401 or name == "LocalTokenNotFoundError":
        return "Hugging Face rejected the saved token. Save a valid token in Settings."
    if status == 403:
        return (
            "The saved Hugging Face token cannot write here. "
            "Create a token with write access and save it in Settings."
        )
    if status == 404 or name in {"RepositoryNotFoundError", "RevisionNotFoundError"}:
        return "That Hugging Face repository was not found, or the saved token cannot read it."
    if name in {
        "ConnectionError",
        "ConnectError",
        "Timeout",
        "ReadTimeout",
        "OfflineModeIsEnabled",
    }:
        return "Could not reach Hugging Face. Check the connection and try again."
    return f"Hugging Face transfer failed ({name})."


def push(api, request):
    kind, repo, folder = request["repo_type"], request["repo_id"], Path(request["folder"])
    api.create_repo(repo, repo_type=kind, private=request["private"], exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        repo_type=kind,
        folder_path=str(folder),
        commit_message=request["message"],
        ignore_patterns=IGNORED,
    )
    for path, text in sorted(request.get("files", {}).items()):
        api.upload_file(
            path_or_fileobj=text.encode(),
            path_in_repo=path,
            repo_id=repo,
            repo_type=kind,
            commit_message=f"Add {path}",
        )
    info = api.repo_info(repo, repo_type=kind)
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
        request = json.loads(sys.stdin.buffer.read(1024**2))
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        result = (push if request["operation"] == "push" else pull)(api, request)
    except Exception as error:
        result = {"error": explain(error)}
    print(json.dumps(result))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
