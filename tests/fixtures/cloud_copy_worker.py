"""Offline process fixture: real copy algorithm over generation-pinned local objects."""

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from vla_platform.lifecycle import cloud_materialize


class Blob:
    generation = 7

    def __init__(self, path):
        self.path = path
        self.size = path.stat().st_size

    def download_as_bytes(self):
        return self.path.read_bytes()

    def open(self, mode, **kwargs):
        assert mode == "rb" and kwargs["raw_download"]
        if os.environ.get("FIREBIRD_TEST_COPY_HANG"):
            Path(os.environ["FIREBIRD_TEST_COPY_HANG"]).write_text(str(os.getpid()))
            time.sleep(300)
        return self.path.open("rb")


class Bucket:
    def get_blob(self, name):
        path = Path(os.environ["FIREBIRD_TEST_GCS_STORE"]) / name
        return Blob(path) if path.is_file() else None


cloud_materialize.cloud_storage.client = lambda: SimpleNamespace(bucket=lambda _: Bucket())
cloud_materialize.copy_remote(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4])
