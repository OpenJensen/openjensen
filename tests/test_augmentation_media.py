import asyncio
import shutil
import subprocess

import pytest
from vla_platform.augmentation.media import duration, trim, verify_output

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is an optional augmentation runtime dependency",
)


def test_real_video_trim_uses_camera_offset_and_checks_timing(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=96x64:r=20:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=96x64:r=20:d=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    original_bytes = source.read_bytes()
    target = tmp_path / "trim.mp4"

    async def run():
        await trim(source, target, 1, 1)
        assert abs(await duration(target) - 1) < 0.05
        await verify_output(target, 1)
        with pytest.raises(ValueError, match="changed the clip duration"):
            await verify_output(target, 2)
        with pytest.raises(ValueError, match="cover the requested"):
            await trim(source, tmp_path / "too-short.mp4", 1.5, 1)

    asyncio.run(run())
    # A real decoded frame must be blue (episode offset), not the earlier red episode.
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(target),
            "-frames:v",
            "1",
            "-vf",
            "scale=1:1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert raw[2] > 200 and raw[0] < 30
    assert source.read_bytes() == original_bytes


def test_invalid_generated_video_is_rejected(tmp_path):
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="Video processing failed"):
        asyncio.run(verify_output(invalid, 5))


def test_playlist_disguised_as_mp4_cannot_load_local_media(tmp_path):
    private = tmp_path / "private.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=96x64:r=20:d=1",
            "-c:v",
            "libx264",
            str(private),
        ],
        check=True,
        capture_output=True,
    )
    playlist = tmp_path / "source.mp4"
    playlist.write_text(f"ffconcat version 1.0\nfile '{private}'\n")
    with pytest.raises(ValueError, match="Video processing failed"):
        asyncio.run(trim(playlist, tmp_path / "output.mp4", 0, 1))
    with pytest.raises(ValueError, match="Video processing failed"):
        asyncio.run(duration(playlist))


def test_ntsc_clip_over_ten_seconds_fails_before_upload(tmp_path):
    source = tmp_path / "ntsc.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=96x64:r=30000/1001:d=11",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="at most 10 seconds"):
        asyncio.run(trim(source, tmp_path / "ten-seconds.mp4", 0, 10))
