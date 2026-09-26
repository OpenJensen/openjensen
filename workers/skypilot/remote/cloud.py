"""GCE identity and create-only diagnostics storage."""

import argparse
import ipaddress
import json
import mimetypes
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen


_METADATA = "http://metadata.google.internal/computeMetadata/v1/"
_API_TIMEOUT = 60
_RETRY_DELAYS = (1, 3, 10)
_LIFETIME_HOURS = 48
_CREATE_ONLY = 0
_POLICY_POLL_SECONDS = 5
_ROLLOUT_FILES = {"worker.log", "container.log", "cleanup.log", "result.json",
                  "result.pending.json", "trajectory.jsonl", "video.mp4", "final.ppm",
                  "job-result.json"}


def _send(request):
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            with urlopen(request, timeout=_API_TIMEOUT) as response:
                return response.read()
        except HTTPError as error:
            retryable = error.code == HTTPStatus.TOO_MANY_REQUESTS or error.code >= HTTPStatus.INTERNAL_SERVER_ERROR
            if not retryable or attempt == len(_RETRY_DELAYS):
                raise
        except URLError:
            if attempt == len(_RETRY_DELAYS):
                raise
        time.sleep(_RETRY_DELAYS[attempt])
    raise RuntimeError("Unreachable retry state")


def _metadata(path):
    request = Request(_METADATA + path, headers={"Metadata-Flavor": "Google"})
    return _send(request).decode()


def _token():
    return json.loads(_metadata("instance/service-accounts/default/token"))["access_token"]


def _instance_url():
    project = _metadata("project/project-id")
    zone = _metadata("instance/zone").rsplit("/", 1)[-1]
    name = _metadata("instance/name")
    return f"https://compute.googleapis.com/compute/v1/projects/{project}/zones/{zone}/instances/{name}"


def _compute(method):
    request = Request(_instance_url(), method=method,
                      headers={"Authorization": "Bearer " + _token()})
    return json.loads(_send(request))


def _deadline():
    created = _compute("GET")["creationTimestamp"]
    deadline = datetime.fromisoformat(created.replace("Z", "+00:00")) + timedelta(hours=_LIFETIME_HOURS)
    return deadline.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _delete():
    # Only this VM's metadata determines the deletion target.
    result = _compute("DELETE")
    if result.get("error"):
        raise RuntimeError(f"VM deletion rejected: {result['error']}")
    print("Host lifetime watchdog requested VM deletion.", flush=True)


def _login(image):
    registry = image.split("/", 1)[0]
    subprocess.run(["docker", "login", registry, "--username", "oauth2accesstoken", "--password-stdin"],
                   input=_token() + "\n", text=True, check=True, timeout=_API_TIMEOUT)


def _upload(path, destination):
    target = urlsplit(destination)
    if target.scheme != "gs" or not target.netloc or target.query or target.fragment:
        raise ValueError("SIM_RESULTS_URI must be a gs:// bucket prefix")
    name = "/".join(filter(None, (target.path.strip("/"), path.name)))
    query = urlencode({"uploadType": "media", "name": name, "ifGenerationMatch": _CREATE_ONLY})
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{quote(target.netloc, safe='')}/o?{query}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    request = Request(url, data=path.read_bytes(), method="POST", headers={
        "Authorization": "Bearer " + _token(), "Content-Type": content_type,
    })
    # UUID prefixes prevent collisions. A 412 remains an error, never a false success.
    _send(request)


def _publish(directory, destination):
    # Completion follows durable logs; it must never precede them.
    for path in sorted(directory.glob("*.log")):
        _upload(path, destination)
    _upload(directory / "job-result.json", destination)
    print(f"Job report: {destination}/job-result.json", flush=True)


def _publish_tree(directory, destination):
    root = directory.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Rollout publication refuses symlinks")
    for path in files:
        if path.name not in _ROLLOUT_FILES:
            raise ValueError(f"Unexpected rollout artifact: {path.name}")
    if not (root / "job-result.json").is_file():
        raise ValueError("Rollout job-result.json is missing")
    # Publish completion only after every trace, video and diagnostic is durable.
    files.sort(key=lambda path: path == root / "job-result.json")
    for path in files:
        parent = path.parent.relative_to(root).as_posix()
        prefix = destination.rstrip("/")
        if parent != ".":
            prefix += "/" + parent
        _upload(path, prefix)
    print(f"Rollout artifacts: {destination}", flush=True)


def _policy_addresses(project, run_id, network):
    query = {
        "filter": f"labels.rollout-id = {run_id} AND labels.rollout-role = vla AND status = RUNNING"
    }
    addresses = []
    while True:
        url = (
            f"https://compute.googleapis.com/compute/v1/projects/{quote(project, safe='')}"
            f"/aggregated/instances?{urlencode(query)}"
        )
        response = json.loads(_send(Request(url, headers={"Authorization": "Bearer " + _token()})))
        for scope in response.get("items", {}).values():
            for instance in scope.get("instances", []):
                for interface in instance.get("networkInterfaces", []):
                    if interface.get("network") == network:
                        addresses.append(interface["networkIP"])
        if not response.get("nextPageToken"):
            return addresses
        query["pageToken"] = response["nextPageToken"]


def _discover_policy(run_id, timeout):
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("ROLLOUT_ID must be the launcher's UUID hex label")
    if not 0 < timeout <= 3600:
        raise ValueError("Policy discovery timeout must be within one hour")
    project = _metadata("project/project-id")
    network = _compute("GET")["networkInterfaces"][0]["network"]
    deadline = time.monotonic() + timeout
    while True:
        addresses = _policy_addresses(project, run_id, network)
        if len(addresses) > 1:
            raise ValueError("Multiple policy VMs match this rollout; refusing ambiguous discovery")
        if addresses:
            address = ipaddress.ip_address(addresses[0])
            if not address.is_private or address.is_loopback:
                raise ValueError("Policy VM must have a private VPC address")
            return str(address)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("No running policy VM found for this rollout")
        time.sleep(min(_POLICY_POLL_SECONDS, remaining))


def _main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("deadline")
    commands.add_parser("delete")
    commands.add_parser("login").add_argument("image")
    publication = commands.add_parser("publish-tree")
    publication.add_argument("directory", type=Path)
    publication.add_argument("destination")
    discovery = commands.add_parser("discover-policy")
    discovery.add_argument("run_id")
    discovery.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.action == "deadline":
        print(_deadline())
        return
    if args.action == "login":
        _login(args.image)
        return
    if args.action == "publish-tree":
        _publish_tree(args.directory, args.destination)
        return
    if args.action == "discover-policy":
        print(_discover_policy(args.run_id, args.timeout))
        return
    _delete()


if __name__ == "__main__":
    _main()
