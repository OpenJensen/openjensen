import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit


_REMOTE = Path(__file__).resolve().parents[1] / "remote"
_IMAGE = "us-east4-docker.pkg.dev/project/simulation/worker@sha256:" + "a" * 64


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cloud = _load("sky_remote_cloud", _REMOTE / "cloud.py")
with patch.dict(sys.modules, {"cloud": cloud}):
    runner = _load("sky_remote_runner", _REMOTE / "runner.py")
preflight = _load("sky_remote_preflight", _REMOTE / "preflight.py")


def _success(root):
    output = root / "outputs" / "worker-id"
    output.mkdir(parents=True)
    video = output / "video.mp4"
    video.write_bytes(b"test-video-content")
    record = {"status": "succeeded", "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
              "video_uri": "gs://results/runs/worker-id/video.mp4"}
    (output / "result.json").write_text(json.dumps(record))
    return output


class CloudTests(unittest.TestCase):
    def test_creation_deadline(self):
        instance = {"creationTimestamp": "2026-09-26T05:00:00-07:00"}
        with patch.object(cloud, "_compute", return_value=instance) as compute:
            self.assertEqual(cloud._deadline(), "2026-09-28 12:00:00 UTC")
        compute.assert_called_once_with("GET")

    def test_login_token_stdin(self):
        with patch.object(cloud, "_token", return_value="secret-token"), \
             patch.object(cloud.subprocess, "run") as run:
            cloud._login(_IMAGE)
        self.assertNotIn("secret-token", " ".join(run.call_args.args[0]))
        self.assertEqual(run.call_args.kwargs["input"], "secret-token\n")

    def test_upload_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.log"
            path.write_text("logs")
            with patch.object(cloud, "_token", return_value="token"), \
                 patch.object(cloud, "_send") as send:
                cloud._upload(path, "gs://results/job/unique-id")
        request = send.call_args.args[0]
        query = parse_qs(urlsplit(request.full_url).query)
        self.assertEqual(query["ifGenerationMatch"], ["0"])
        self.assertEqual(query["name"], ["job/unique-id/worker.log"])

    def test_upload_conflict_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.log"
            path.write_text("new logs")
            error = HTTPError("https://storage.googleapis.com", 412, "exists", {}, None)
            with patch.object(cloud, "_token", return_value="token"), \
                 patch.object(cloud, "_send", side_effect=error), \
                 self.assertRaises(HTTPError):
                cloud._upload(path, "gs://results/unique-id")

    def test_report_published_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "worker.log").write_text("log")
            (root / "job-result.json").write_text("{}")
            with patch.object(cloud, "_upload") as upload:
                cloud._publish(root, "gs://results/unique-id")
        self.assertEqual(upload.call_args_list[-1].args[0].name, "job-result.json")


class RunnerTests(unittest.TestCase):
    def test_zero_without_result(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            with patch.object(cloud, "_publish") as publish:
                code = runner._finish(state, {"exit_code": 0}, "gs://results/job")
            record = json.loads((state / "job-result.json").read_text())
        self.assertEqual(code, os.EX_DATAERR)
        self.assertEqual(record["status"], "failed")
        publish.assert_called_once()

    def test_missing_video_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            output = _success(state)
            (output / "video.mp4").unlink()
            with self.assertRaisesRegex(ValueError, "nonempty video"):
                runner._validate(state / "outputs")

    def test_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            output = _success(state)
            (output / "video.mp4").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                runner._validate(state / "outputs")

    def test_valid_output_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            _success(state)
            with patch.object(cloud, "_publish"):
                code = runner._finish(state, {"exit_code": 0}, "gs://results/job")
        self.assertEqual(code, 0)

    def test_python310_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            _success(state)
            with patch.object(runner.hashlib, "file_digest", None):
                self.assertEqual(runner._validate(state / "outputs")["status"], "succeeded")

    def test_malformed_result_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            output = _success(state)
            (output / "result.json").write_text("[]")
            with patch.object(cloud, "_publish"):
                code = runner._finish(state, {"exit_code": 0}, "gs://results/job")
        self.assertEqual(code, os.EX_DATAERR)

    def test_missing_video_uri_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            output = _success(state)
            record = json.loads((output / "result.json").read_text())
            del record["video_uri"]
            (output / "result.json").write_text(json.dumps(record))
            with patch.object(cloud, "_publish"):
                code = runner._finish(state, {"exit_code": 0}, "gs://results/job")
        self.assertEqual(code, os.EX_DATAERR)

    def test_original_error_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(cloud, "_publish", side_effect=RuntimeError("offline")):
                code = runner._finish(Path(directory), {"exit_code": 124}, "gs://results/job")
        self.assertEqual(code, 124)

    def test_timeout_stops_client(self):
        process = Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner.subprocess, "Popen", return_value=process), \
                 patch.object(runner.time, "monotonic", side_effect=[0, 1000]), \
                 self.assertRaises(subprocess.TimeoutExpired):
                runner._capture(["docker", "run"], Path(directory) / "worker.log", 1)
        process.terminate.assert_called_once()
        process.wait.assert_called_once()

    def test_stop_failure_kills(self):
        failed = subprocess.CompletedProcess([], 1)
        success = subprocess.CompletedProcess([], 0)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner.subprocess, "run", side_effect=[failed, success, success, success]) as run:
                runner._cleanup("container", Path(directory))
        self.assertEqual(run.call_args_list[1].args[0], ["docker", "kill", "container"])
        self.assertEqual(run.call_args_list[-1].args[0], ["docker", "rm", "-f", "container"])

    def test_source_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir(mode=0o700)
            (source / "worker.py").write_text("original")
            first = runner._snapshot(source, root / "first")
            (source / "worker.py").write_text("changed")
            second = runner._snapshot(source, root / "second")
            self.assertEqual((root / "first" / "worker.py").read_text(), "original")
            self.assertEqual((root / "first").stat().st_mode & 0o777, 0o755)
            self.assertNotEqual(first, second)


class SetupTests(unittest.TestCase):
    def test_install_missing_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = root / "command"
            # Shell quoting keeps the fixture executable when Python's path has spaces.
            command.write_text(
                '#!/bin/sh\n'
                'exec "$SIM_TEST_PYTHON" "$SIM_TEST_HANDLER" "$0" "$@"\n'
            )
            handler = root / "handler.py"
            handler.write_text('''
import json, os, pathlib, sys

root = pathlib.Path(os.environ["SIM_TEST_ROOT"])
name = pathlib.Path(sys.argv[1]).name
args = sys.argv[2:]
with (root / "calls").open("a") as stream:
    stream.write(json.dumps([name, *args]) + "\\n")
if name == "dpkg-query":
    sys.exit(1)
if name == "sudo":
    if args[0] == "tee":
        sys.stdin.read()
    if "apt-get" in args and "install" in args and "libvulkan1" in args:
        (root / "loader").touch()
elif name == "python3":
    if args[-1] == "deadline":
        print("2026-09-28 12:58:21 UTC")
    if args[0].endswith("preflight.py") and not (root / "loader").exists():
        sys.exit("libvulkan.so.1: cannot open shared object file")
''')
            command.chmod(0o755)
            for name in ("sudo", "python3", "docker", "timeout", "dpkg-query"):
                (root / name).symlink_to(command)
            environment = os.environ | {
                "PATH": f"{root}:{os.environ['PATH']}",
                "SIM_TEST_ROOT": str(root),
                "SIM_TEST_PYTHON": sys.executable,
                "SIM_TEST_HANDLER": str(handler),
                "SIM_IMAGE": _IMAGE,
            }
            result = subprocess.run(
                ["bash", str(_REMOTE / "setup.sh")], env=environment,
                text=True, capture_output=True, check=False, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in (root / "calls").read_text().splitlines()]
        installs = [call for call in calls if "apt-get" in call and "install" in call]
        self.assertEqual(len(installs), 1)
        self.assertIn("libvulkan1", installs[0])
        self.assertIn("--no-install-recommends", installs[0])


class PreflightTests(unittest.TestCase):
    def _host(self, root):
        libraries = {name: root / name for name in (
            "libGLX_nvidia.so.0", "libEGL_nvidia.so.0", "libvulkan.so.1",
        )}
        for path in libraries.values():
            path.write_bytes(b"test library")
        (root / "nvidia_icd.json").write_text(json.dumps({
            "ICD": {"library_path": "libGLX_nvidia.so.0"},
        }))
        return libraries

    def _response(self, command, libraries):
        name = Path(command[0]).name
        output = ""
        if name == "nvidia-smi":
            output = "580.159.04, NVIDIA L4\n"
        elif name == "ldconfig":
            output = "libGLX_nvidia.so.0 (libc6) => /wrong/32bit/library\n"
            output += "\n".join(f"{name} (libc6,x86-64) => {path}"
                                for name, path in libraries.items())
        elif name == "ldd":
            output = "libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x1234)\n"
        elif name != "docker":
            raise AssertionError(f"Unexpected command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    def test_no_native_driver_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = self._host(root)
            with patch.object(preflight, "_ICD_DIRS", (root,)), \
                 patch.object(preflight.subprocess, "run", side_effect=lambda cmd, **kw: self._response(cmd, libraries)) as run, \
                 patch("ctypes.CDLL", side_effect=AssertionError("Driver loaded into Python")):
                preflight._check()

        # Inspect dependencies without starting native driver lifecycle in Python.
        checks = [call for call in run.call_args_list if Path(call.args[0][0]).name == "ldd"]
        self.assertEqual({call.args[0][-1] for call in checks}, {str(path) for path in libraries.values()})
        self.assertTrue(all(call.kwargs.get("timeout", 0) > 0 for call in checks))

    def test_missing_dependency_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = self._host(root)

            def response(command, **kwargs):
                result = self._response(command, libraries)
                if Path(command[0]).name == "ldd":
                    result.stdout = "libnvidia-dependency.so.1 => not found\n"
                return result

            with patch.object(preflight, "_ICD_DIRS", (root,)), \
                 patch.object(preflight.subprocess, "run", side_effect=response), \
                 patch("ctypes.CDLL", side_effect=AssertionError("Driver loaded into Python")), \
                 self.assertRaisesRegex(RuntimeError, "libnvidia-dependency.so.1"):
                preflight._check()

    def test_missing_library_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = self._host(root)
            del libraries["libvulkan.so.1"]
            with patch.object(preflight, "_ICD_DIRS", (root,)), \
                 patch.object(preflight.subprocess, "run", side_effect=lambda cmd, **kw: self._response(cmd, libraries)), \
                 patch("ctypes.CDLL", side_effect=AssertionError("Driver loaded into Python")), \
                 self.assertRaisesRegex(RuntimeError, "libvulkan.so.1"):
                preflight._check()

    def test_absolute_icd_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = self._host(root)
            icd = root / "libnvidia-icd.so"
            icd.write_bytes(b"test ICD library")
            (root / "nvidia_icd.json").write_text(json.dumps({"ICD": {"library_path": str(icd)}}))
            with patch.object(preflight, "_ICD_DIRS", (root,)), \
                 patch.object(preflight.subprocess, "run", side_effect=lambda cmd, **kw: self._response(cmd, libraries)) as run, \
                 patch("ctypes.CDLL", side_effect=AssertionError("Driver loaded into Python")):
                preflight._check()

        self.assertIn(str(icd), [call.args[0][-1] for call in run.call_args_list
                                if Path(call.args[0][0]).name == "ldd"])

    def test_old_driver_rejected(self):
        response = subprocess.CompletedProcess([], 0, stdout="535.1, NVIDIA L4\n")
        with patch.object(preflight.subprocess, "run", return_value=response), \
             self.assertRaisesRegex(RuntimeError, "older than"):
            preflight._check()

    def test_graphics_required(self):
        response = subprocess.CompletedProcess([], 0, stdout="580.105.08, NVIDIA L4\n")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(preflight.subprocess, "run", return_value=response), \
                 patch.object(preflight, "_ICD_DIRS", (Path(directory),)), \
                 self.assertRaisesRegex(RuntimeError, "Vulkan ICD"):
                preflight._check()


if __name__ == "__main__":
    unittest.main()
