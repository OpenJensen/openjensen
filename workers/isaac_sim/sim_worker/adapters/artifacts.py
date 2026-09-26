import hashlib
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from sim_worker.contracts import API_VERSION, RunSpec, Status


_CREATE_ONLY = 0
_UPLOAD_TIMEOUT_SECONDS = 120


class Artifacts:
    def __init__(self, root: Path, spec: RunSpec):
        self._spec = spec
        self._run_id = str(uuid.uuid4())
        self._directory = root / self._run_id
        self._directory.mkdir(parents=True, exist_ok=False)
        self._uri = f"{spec.output_uri}/{self._run_id}"
        self._started = datetime.now(timezone.utc).isoformat()
        self._client = None

    @property
    def video_path(self) -> Path:
        return self._directory / "video.mp4"

    def start(self) -> None:
        # Check upload credentials before spending GPU time on the simulation.
        path = self._directory / "manifest.json"
        self._write(path, {"api_version": API_VERSION, "spec": asdict(self._spec)})
        self._upload(path, "application/json")

    def complete(self) -> dict:
        with self.video_path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        self._upload(self.video_path, "video/mp4")
        result = self._record(Status.SUCCEEDED)
        result.update({
            "video_uri": f"{self._uri}/video.mp4", "sha256": digest,
            "capture": {"width": self._spec.width, "height": self._spec.height,
                        "fps": self._spec.fps, "frames": self._spec.frames,
                        "duration_seconds": self._spec.frames / self._spec.fps},
        })
        # Publish completion last; consumers must wait for this record.
        path = self._directory / "result.json"
        self._write(path, result)
        self._upload(path, "application/json")
        return result

    def fail(self, error: str) -> None:
        result = self._record(Status.FAILED)
        result["error"] = error
        path = self._directory / "result.json"
        self._write(path, result)
        self._upload(path, "application/json")

    def _record(self, status: Status) -> dict:
        return {
            "api_version": API_VERSION, "run_id": self._run_id, "status": status.value,
            "started_at": self._started, "finished_at": datetime.now(timezone.utc).isoformat(),
            "result_uri": f"{self._uri}/result.json",
        }

    def _write(self, path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def _upload(self, path: Path, content_type: str) -> None:
        from google.cloud import storage

        if self._client is None:
            self._client = storage.Client()
        target = urlsplit(self._uri)
        name = f"{target.path.strip('/')}/{path.name}"
        blob = self._client.bucket(target.netloc).blob(name)
        blob.upload_from_filename(
            str(path), content_type=content_type, if_generation_match=_CREATE_ONLY,
            timeout=_UPLOAD_TIMEOUT_SECONDS, checksum="auto",
        )
