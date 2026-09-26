"""Keep probe evidence separate across repeated runs."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


_TASK = Path(__file__).resolve().parents[1] / "probe.example.yaml"


class ProbeTests(unittest.TestCase):
    def test_runs_keep_separate_output(self):
        task = yaml.safe_load(_TASK.read_text())
        runs = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = {
                "mkdir": "exit 0\n",
                "sudo": "exit 0\n",
                "python3": 'printf "%s\\n" "$SIM_TEST_RUN_ID"\n',
                "docker": 'printf "%s\\n" "$@" > "$SIM_TEST_CALLS.$1"\n',
            }
            for name, body in scripts.items():
                script = root / name
                script.write_text("#!/bin/bash\n" + body)
                script.chmod(0o755)

            for run_id in ("first-run", "second-run"):
                environment = os.environ | {
                    "PATH": str(root) + os.pathsep + os.environ["PATH"],
                    "ACCEPT_EULA": "Y",
                    "SIM_PROBE_LABEL": "realtime",
                    "SIM_IMAGE": "test-image",
                    "SIM_TEST_RUN_ID": run_id,
                    "SIM_TEST_CALLS": str(root / "calls"),
                }
                result = subprocess.run(
                    ["bash", "-c", task["run"]], env=environment,
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                args = (root / "calls.run").read_text().splitlines()
                mount = next(arg for arg in args if arg.endswith(",dst=/probe-output"))
                output = mount.removeprefix("type=bind,src=").removesuffix(",dst=/probe-output")
                container = args[args.index("--name") + 1]
                runs.append((output, container, result.stdout, run_id))
                self.assertEqual((root / "calls.rm").read_text().splitlines(), ["rm", "-f", container])

        self.assertNotEqual(runs[0][0], runs[1][0])
        for output, container, stdout, run_id in runs:
            self.assertTrue(output.endswith(run_id))
            self.assertEqual(container, "isaac-probe-" + run_id)
            self.assertIn(output, stdout)


if __name__ == "__main__":
    unittest.main()
