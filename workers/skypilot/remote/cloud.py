"""GCE identity and create-only diagnostics storage."""

import argparse
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
import json
import mimetypes
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen


_METADATA = "http://metadata.google.internal/computeMetadata/v1/"
_API_TIMEOUT = 60
_RETRY_DELAYS = (1, 3, 10)
_LIFETIME_HOURS = 48
_CREATE_ONLY = 0


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


def _main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("deadline")
    commands.add_parser("delete")
    commands.add_parser("login").add_argument("image")
    args = parser.parse_args()
    if args.action == "deadline":
        print(_deadline())
        return
    if args.action == "login":
        _login(args.image)
        return
    _delete()


if __name__ == "__main__":
    _main()
