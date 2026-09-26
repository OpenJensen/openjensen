import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


_BUILD = Path(__file__).resolve().parents[1] / "build.sh"
_PROJECT = "example-simulation-project"


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.script = self.root / "build.sh"
        shutil.copy2(_BUILD, self.script)
        self.log = self.root / "cloud-args.txt"
        cloud = self.root / "gcloud"
        cloud.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SIM_TEST_LOG"\n')
        cloud.chmod(0o755)
        self.env = os.environ | {
            "PATH": f"{self.root}:{os.environ['PATH']}",
            "SIM_TEST_LOG": str(self.log),
        }
        self.env.pop("SIM_PROJECT_ID", None)

    def test_project_required(self):
        result = subprocess.run(["bash", str(self.script)], env=self.env,
                                capture_output=True, text=True, timeout=10)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SIM_PROJECT_ID", result.stderr)
        self.assertFalse(self.log.exists())

    def test_project_selected(self):
        self.env["SIM_PROJECT_ID"] = _PROJECT
        subprocess.run(["bash", str(self.script)], env=self.env, check=True,
                       capture_output=True, text=True, timeout=10)

        args = self.log.read_text().splitlines()
        self.assertIn(f"--project={_PROJECT}", args)
        self.assertIn(f"--gcs-source-staging-dir=gs://{_PROJECT}-sim-build-source/source", args)


if __name__ == "__main__":
    unittest.main()
