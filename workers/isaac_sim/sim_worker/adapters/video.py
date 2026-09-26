import json
import subprocess
from fractions import Fraction
from pathlib import Path

from sim_worker.contracts import RunSpec


_ENCODE_TIMEOUT_SECONDS = 120
_PROBE_TIMEOUT_SECONDS = 30
_H264_QUALITY = 20
_ENCODER_THREADS = 2


class Recorder:
    def __init__(self, path: Path, spec: RunSpec):
        self._path = path
        self._spec = spec
        self._process = None
        self._count = 0

    def __enter__(self):
        # Stream frames through a bounded pipe; never retain the whole video in RAM.
        self._process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{self._spec.width}x{self._spec.height}",
            "-framerate", str(self._spec.fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-threads", str(_ENCODER_THREADS),
            "-preset", "fast", "-crf", str(_H264_QUALITY),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self._path),
        ], stdin=subprocess.PIPE)
        return self

    def append(self, frame: bytes) -> None:
        if len(frame) != self._spec.frame_bytes:
            raise ValueError("Frame byte count does not match the manifest")
        if self._count >= self._spec.frames:
            raise ValueError("Simulator produced too many frames")
        self._process.stdin.write(frame)
        self._count += 1

    def __exit__(self, error_type, error, traceback):
        if error_type is not None:
            self._abort()
            return

        try:
            self._process.stdin.close()
            code = self._process.wait(timeout=_ENCODE_TIMEOUT_SECONDS)
            if code != 0 or self._count != self._spec.frames:
                raise RuntimeError(f"Incomplete recording: encoder={code}, frames={self._count}")
            self._verify()
        except BaseException:
            self._abort()
            raise

    def _abort(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        self._path.unlink(missing_ok=True)

    def _verify(self) -> None:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(self._path),
        ], check=True, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_SECONDS)
        streams = json.loads(result.stdout)["streams"]
        if len(streams) != 1:
            raise RuntimeError("Recording must contain one video stream")
        stream = streams[0]
        actual = (stream["codec_name"], stream["width"], stream["height"],
                  Fraction(stream["avg_frame_rate"]), int(stream["nb_read_frames"]))
        expected = ("h264", self._spec.width, self._spec.height, self._spec.fps, self._spec.frames)
        if actual != expected:
            raise RuntimeError(f"Recording metadata mismatch: {actual}")
