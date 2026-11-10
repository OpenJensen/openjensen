"""Explicit fake-bus companion for collection verification; never opens USB."""

import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from server import RobotLab, create_server

# Stateful fake transport shared with the protocol's safety regression tests.
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from test_robot_lab import CALIBRATION, FakeBus


class SyntheticCamera:
    def read(self, index=0, **kwargs):
        if index not in (0, 1):
            raise ValueError("Synthetic camera missing")
        image = np.zeros((96, 128, 3), dtype=np.uint8)
        image[:, :, index] = 70
        x = int(time.monotonic() * 10) % 90
        image[30:50, x : x + 20] = [240, 240, 240]
        return image

    def start_reader(self, index, **kwargs):
        self.read(index)

    def read_latest_timed(self, index=0, **kwargs):
        return self.read(index), time.monotonic()

    def capture_timed(self, index=0):
        started = time.monotonic_ns()
        stream = io.BytesIO()
        Image.fromarray(self.read(index)[:, :, ::-1]).save(stream, format="JPEG")
        return {
            "jpeg": stream.getvalue(),
            "read_start_monotonic_ns": started,
            "read_end_monotonic_ns": time.monotonic_ns(),
        }

    def latest_timed_jpeg(self, index=0, **kwargs):
        return self.capture_timed(index)

    def jpeg(self, index=0, **kwargs):
        return self.capture_timed(index)["jpeg"]

    cached_jpeg = jpeg

    def close(self):
        pass

    def close_idle(self):
        pass


def main():
    config = json.loads(sys.stdin.readline())
    root = Path(config["root"])
    lab = RobotLab(
        root,
        bus_factory=FakeBus,
        camera=SyntheticCamera(),
        port_provider=lambda: [{"device": "/dev/mock"}],
    )
    lab.calibration_root = root / "calibrations"
    path = lab.calibration_path("mock-so101", "follower")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(CALIBRATION))
    lab.control_start(
        {
            "port": "/dev/mock",
            "robot_id": "mock-so101",
            "mode": "manual",
            "confirm": "ENABLE MOTION",
        }
    )
    server = create_server(lab, config["token"], port=0)
    print(json.dumps({"port": server.server_address[1], "generation": lab.generation}), flush=True)

    def parent_closed():
        sys.stdin.read()
        server.shutdown()

    threading.Thread(target=parent_closed, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        lab.close()


if __name__ == "__main__":
    main()
