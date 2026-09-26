"""Auth sequencing checks using stubs; no service-account key is read."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PROJECT = "project-5693e83a-db3a-43e1-98c"
_ACCOUNT = f"sim-rollout-runner@{_PROJECT}.iam.gserviceaccount.com"
_GCLOUD_STUB = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$SIM_AUTH_TEST_LOG"
case "$1 $2" in
  'config configurations')
    if [[ "$3" != list || -n "${CLOUDSDK_ACTIVE_CONFIG_NAME:-}" ]]; then exit 2; fi
    mkdir -p "$CLOUDSDK_CONFIG/configurations"
    : > "$CLOUDSDK_CONFIG/configurations/config_default"
    ;;
  'auth activate-service-account')
    if [[ "${CLOUDSDK_ACTIVE_CONFIG_NAME:-}" == default &&
          ! -f "$CLOUDSDK_CONFIG/configurations/config_default" ]]; then
      echo 'Cannot set property in configuration [default], it does not exist.' >&2
      exit 42
    fi
    if [[ "${SIM_AUTH_TEST_FAIL:-}" == activation ]]; then exit 43; fi
    ;;
  'config set') ;;
  'auth list')
    if [[ "${SIM_AUTH_TEST_FAIL:-}" == identity ]]; then
      printf 'unexpected-account\n'
    else
      printf '%s\n' "$CLOUDSDK_CORE_ACCOUNT"
    fi
    ;;
  *) exit 2 ;;
esac
"""


class AuthTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="auth-test-", dir=_ROOT / "tests")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        for name, contents in (
            ("gcloud", _GCLOUD_STUB),
            ("python-stub", "#!/usr/bin/env bash\nprintf '/unused/dedicated-runner-key.json\\n'\n"),
        ):
            path = self.bin / name
            path.write_text(contents)
            path.chmod(0o700)
        self.environment = dict(os.environ)
        self.environment.update(
            PATH=f"{self.bin}:{os.environ['PATH']}",
            SIM_PYTHON=str(self.bin / "python-stub"),
            SIM_AUTH_STATE_DIR=str(self.directory / "isolated"),
            SIM_AUTH_TEST_LOG=str(self.directory / "calls.log"),
            CLOUDSDK_ACTIVE_CONFIG_NAME="personal-profile",
            CLOUDSDK_CORE_ACCOUNT="previous-account",
            SIM_PROJECT_ID="previous-project",
        )

    def _source(self, key="/unused/dedicated-runner-key.json"):
        command = r"""
source "$1" "$2"
sim_result=$?
printf 'STATUS=%s\nPROJECT=%s\nACTIVE=%s\nACCOUNT=%s\n' \
  "$sim_result" "${SIM_PROJECT_ID:-unset}" \
  "${CLOUDSDK_ACTIVE_CONFIG_NAME:-unset}" "${CLOUDSDK_CORE_ACCOUNT:-unset}"
"""
        return subprocess.run(
            ["bash", "-c", command, "auth-test", str(_ROOT / "auth.sh"), str(key)],
            env=self.environment,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_fresh_profile(self):
        result = self._source()
        self.assertIn("STATUS=0", result.stdout, result.stderr)
        self.assertIn(f"PROJECT={_PROJECT}", result.stdout)
        self.assertIn("ACTIVE=default", result.stdout)
        self.assertIn(f"ACCOUNT={_ACCOUNT}", result.stdout)
        calls = (self.directory / "calls.log").read_text().splitlines()
        self.assertTrue(calls[0].startswith("config configurations list"))
        self.assertTrue(calls[1].startswith("auth activate-service-account"))

    def test_repo_key_is_rejected(self):
        # Reject keys anywhere in the checkout, including outside this worker.
        with tempfile.TemporaryDirectory(dir=_ROOT.parent) as directory:
            key = Path(directory) / "key.json"
            key.write_text(json.dumps({
                "type": "service_account",
                "client_email": _ACCOUNT,
                "project_id": _PROJECT,
                "private_key": "test-only-placeholder",
            }))
            self.environment["SIM_PYTHON"] = sys.executable
            result = self._source(key)
        self.assertIn("STATUS=1", result.stdout)
        self.assertIn("outside this repository", result.stderr)
        self.assertFalse((self.directory / "calls.log").exists())

    def test_auth_failure_blocks_sky(self):
        self.environment["SIM_AUTH_TEST_FAIL"] = "activation"
        result = self._source()
        self.assertIn("STATUS=1", result.stdout)
        self.assertIn("PROJECT=unset", result.stdout)
        calls = (self.directory / "calls.log").read_text()
        self.assertNotIn("config set project", calls)

    def test_wrong_identity_blocks_sky(self):
        self.environment["SIM_AUTH_TEST_FAIL"] = "identity"
        result = self._source()
        self.assertIn("STATUS=1", result.stdout)
        self.assertIn("PROJECT=unset", result.stdout)
        self.assertIn("not confirmed", result.stderr)

    def test_existing_profile_survives(self):
        config = self.directory / "isolated/gcloud/configurations/config_default"
        config.parent.mkdir(parents=True)
        config.write_text("[core]\nproject = existing-dedicated-project\n")
        result = self._source()
        self.assertIn("STATUS=0", result.stdout, result.stderr)
        calls = (self.directory / "calls.log").read_text()
        self.assertNotIn("config configurations list", calls)
        self.assertIn("existing-dedicated-project", config.read_text())


if __name__ == "__main__":
    unittest.main()
