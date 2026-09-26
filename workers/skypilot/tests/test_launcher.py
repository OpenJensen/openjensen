"""Check local configuration failures before dispatching cloud commands."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_PROJECT = "simulation-test-project"


class LauncherTests(unittest.TestCase):
    def _run(self, args, files=None, missing=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("sky.sh", "launch.sh"):
                shutil.copy(_ROOT / name, root / name)
            binaries = root / ".venv" / "bin"
            binaries.mkdir(parents=True)
            for name in ("python", "sky", "gcloud"):
                script = binaries / name
                script.write_text(
                    '#!/usr/bin/env bash\n'
                    'if [[ "${0##*/}" == python ]]; then exit 0; fi\n'
                    'printf "%s\\n" "$@" > "$SIM_TEST_CALLS"\n'
                )
                script.chmod(0o755)
            for name, content in (files or {}).items():
                (root / name).write_text(content)
            calls = root / "calls"
            environment = os.environ | {
                "SIM_PROJECT_ID": _PROJECT,
                "SIM_TEST_CALLS": str(calls),
            }
            if missing:
                environment.pop(missing, None)
            result = subprocess.run(
                ["bash", str(root / "sky.sh"), *args], env=environment,
                capture_output=True, text=True, check=False,
            )
            return result, calls.read_text() if calls.exists() else None

    def test_missing_project_stops(self):
        result, calls = self._run(["status"], missing="SIM_PROJECT_ID")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SIM_PROJECT_ID", result.stderr)
        self.assertIsNone(calls)

    def test_missing_config_stops(self):
        result, calls = self._run(["status"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("config.example.yaml", result.stderr)
        self.assertIsNone(calls)

    def test_placeholder_stops(self):
        files = {"config.yaml": "project: CHANGE_ME_PROJECT_ID\n"}
        result, calls = self._run(["status"], files)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CHANGE_ME", result.stderr)
        self.assertIsNone(calls)

    def test_unedited_task_stops(self):
        files = {"config.yaml": "allowed_clouds: [gcp]\n",
                 "task.yaml": "SIM_IMAGE: CHANGE_ME_IMAGE\n"}
        result, calls = self._run(["launch", "task.yaml"], files)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("task.yaml", result.stderr)
        self.assertIsNone(calls)

    def test_missing_task_stops(self):
        files = {"config.yaml": "allowed_clouds: [gcp]\n"}
        result, calls = self._run(["exec", "isaac-sim", "task.yaml"], files)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("task.yaml", result.stderr)
        self.assertIsNone(calls)

    def test_gcloud_before_config(self):
        result, calls = self._run(["gcloud", "auth", "login", "--update-adc"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, "auth\nlogin\n--update-adc\n")

    def test_manifest_env_dispatches(self):
        files = {"config.yaml": "allowed_clouds: [gcp]\n",
                 "task.yaml": "name: recording\n"}
        args = ["exec", "isaac-sim", "task.yaml", "--env", "SIM_MANIFEST=scene.yaml"]
        result, calls = self._run(args, files)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls.splitlines(), args)

    def test_ready_config_dispatches(self):
        files = {"config.yaml": "allowed_clouds: [gcp]\n"}
        result, calls = self._run(["queue", "isaac-sim"], files)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, "queue\nisaac-sim\n")


if __name__ == "__main__":
    unittest.main()
