"""Exercise the real GCS range-buffer implementation without network/model downloads."""

import hashlib
import math

import pytest
from vla_platform.lifecycle import cloud_storage as storage

fileio = pytest.importorskip("google.cloud.storage.fileio")


def test_real_blobreader_amortizes_ranges_but_keeps_small_output_chunks():
    class TrackingReader(fileio.BlobReader):
        maximum_buffer = 0
        maximum_read = 0

        def read(self, size=-1):
            self.maximum_read = max(self.maximum_read, size)
            block = super().read(size)
            self.maximum_buffer = max(self.maximum_buffer, self._buffer.getbuffer().nbytes)
            return block

    class RangeBlob:
        size = 40 * 1024**2 + 17
        chunk_size = None

        def __init__(self):
            self.ranges = []

        def open(self, mode, **kwargs):
            assert mode == "rb"
            self.reader = TrackingReader(self, **kwargs)
            return self.reader

        def download_as_bytes(self, *, start, end, **kwargs):
            # Match the GCS HTTP range contract: its end byte is inclusive.
            end = min(end, self.size - 1)
            self.ranges.append((start, end))
            pattern = bytes(range(251))
            count, offset = end - start + 1, start % len(pattern)
            return (pattern * math.ceil((count + offset) / len(pattern)))[offset : offset + count]

    blob = RangeBlob()
    digest = hashlib.sha256()
    pattern = bytes(range(251))
    for _ in range(blob.size // len(pattern)):
        digest.update(pattern)
    digest.update(pattern[: blob.size % len(pattern)])
    largest_output, total = 0, 0
    for block in storage.archive_chunks(b"{}", [("model.gguf", blob, digest.hexdigest())]):
        largest_output = max(largest_output, len(block))
        total += len(block)
    assert len(blob.ranges) == math.ceil(blob.size / storage.GCS_READ_AHEAD_BYTES) == 3
    assert max(end - start + 1 for start, end in blob.ranges) <= storage.GCS_READ_AHEAD_BYTES + 1
    assert blob.reader.maximum_buffer <= storage.GCS_READ_AHEAD_BYTES + 1
    assert blob.reader.maximum_read <= 1024**2
    assert largest_output <= 1024**2
    assert total > blob.size
