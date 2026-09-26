"""Exercise IAM and network changes without cloud access."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_PROJECT = "simulation-test-project"
_LAUNCHER = "user:operator@example.com"
_ACCOUNT = f"skypilot-v1@{_PROJECT}.iam.gserviceaccount.com"


class ConfigureTests(unittest.TestCase):
    def _run(self, scenario, missing=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = root / "calls.jsonl"
            self._fake_sky(root)
            self._fake_gcloud(root)
            environment = os.environ | {
                "PATH": f"{root}:{os.environ['PATH']}",
                "PYTHONPATH": str(root),
                "SIM_TEST_CALLS": str(calls),
                "SIM_TEST_SCENARIO": scenario,
                "SIM_TEST_ROLE": str(_ROOT / "role.json"),
                "SIM_PROJECT_ID": _PROJECT,
                "SIM_LAUNCHER": _LAUNCHER,
            }
            if missing:
                environment.pop(missing, None)
            result = subprocess.run(
                ["bash", str(_ROOT / "configure.sh")],
                env=environment, text=True, capture_output=True, check=False,
            )
            commands = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
            return result, commands

    def _fake_sky(self, root):
        package = root / "sky" / "clouds" / "utils"
        package.mkdir(parents=True)
        (package / "gcp_utils.py").write_text(
            "import json, os\n"
            "def get_minimal_compute_permissions():\n"
            "    with open(os.environ['SIM_TEST_ROLE']) as stream:\n"
            "        return json.load(stream)['includedPermissions']\n"
        )
        distribution = root / "skypilot-0.13.0.dist-info"
        distribution.mkdir()
        (distribution / "METADATA").write_text("Name: skypilot\nVersion: 0.13.0\n")

    def _fake_gcloud(self, root):
        script = root / "gcloud"
        script.write_text("""#!/usr/bin/env python3
import json, os, sys

args = sys.argv[1:]
with open(os.environ['SIM_TEST_CALLS'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
scenario = os.environ['SIM_TEST_SCENARIO']
project = os.environ.get('SIM_PROJECT_ID', 'missing-project')
account = f'skypilot-v1@{project}.iam.gserviceaccount.com'
if args[:4] == ['compute', 'networks', 'subnets', 'describe']:
    print(json.dumps({'network': 'global/networks/sim-network',
                      'ipCidrRange': '10.42.0.0/24'}))
elif args[:3] == ['iam', 'roles', 'list']:
    roles = []
    if scenario == 'existing-role':
        roles = [{'name': f'projects/{project}/roles/simSkyCompute'}]
    print(json.dumps(roles))
elif args[:3] == ['iam', 'roles', 'describe']:
    with open(os.environ['SIM_TEST_ROLE']) as stream:
        print(stream.read())
elif args[:2] == ['projects', 'get-iam-policy']:
    bindings = []
    if scenario == 'broad-iam':
        bindings = [{'role': 'roles/owner', 'members': ['serviceAccount:' + account]}]
    print(json.dumps({'bindings': bindings}))
elif args[:3] == ['compute', 'firewall-rules', 'list']:
    if scenario == 'read-failure':
        sys.exit(1)
    rules = []
    if scenario == 'public-ssh':
        rules = [{'network': 'global/networks/sim-network', 'direction': 'INGRESS',
                  'sourceRanges': ['0.0.0.0/0'], 'targetServiceAccounts': [account],
                  'allowed': [{'IPProtocol': 'tcp', 'ports': ['22']}]}]
    print(json.dumps(rules))
elif args[:4] == ['iap', 'tcp', 'dest-groups', 'list']:
    groups = []
    if scenario == 'broad-group':
        groups = [{'name': 'sim-ssh', 'cidrs': ['0.0.0.0/0']}]
    print(json.dumps(groups))
elif args[:3] == ['iam', 'service-accounts', 'list']:
    print('[]')
else:
    print('{}')
""")
        script.chmod(0o755)

    def test_missing_project_stops(self):
        result, commands = self._run("missing", "SIM_PROJECT_ID")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SIM_PROJECT_ID", result.stderr)
        self.assertEqual(commands, [])

    def test_missing_operator_stops(self):
        result, commands = self._run("missing", "SIM_LAUNCHER")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SIM_LAUNCHER", result.stderr)
        self.assertEqual(commands, [])

    def test_scoped_grants(self):
        result, commands = self._run("missing")
        self.assertEqual(result.returncode, 0, result.stderr)
        grants = [args for args in commands if "add-iam-policy-binding" in args]
        project_grants = [args for args in grants if args[0] == "projects"]
        self.assertEqual(len(project_grants), 1)
        self.assertIn(f"--role=projects/{_PROJECT}/roles/simSkyCompute", project_grants[0])
        self.assertEqual(len(grants), 6)
        for args in grants:
            if args[0] == "iap":
                continue
            self.assertIn(f"--member=serviceAccount:{_ACCOUNT}", args)
        firewall = next(args for args in commands if args[:3] == [
            "compute", "firewall-rules", "create"
        ])
        self.assertIn("--source-ranges=35.235.240.0/20", firewall)
        self.assertIn("--rules=tcp:22", firewall)
        self.assertIn(f"--target-service-accounts={_ACCOUNT}", firewall)

    def test_refuse_broad_iam(self):
        self._assert_no_grants("broad-iam")

    def test_operator_iap_grant(self):
        result, commands = self._run("missing")
        self.assertEqual(result.returncode, 0, result.stderr)
        grants = [args for args in commands if args[:4] == [
            "iap", "tcp", "dest-groups", "add-iam-policy-binding"
        ]]
        self.assertEqual(len(grants), 1)
        self.assertIn("--dest-group=sim-ssh", grants[0])
        self.assertIn("--region=us-east4", grants[0])
        self.assertIn(f"--member={_LAUNCHER}", grants[0])
        self.assertIn("--role=roles/iap.tunnelResourceAccessor", grants[0])

    def test_reuse_listed_role(self):
        result, commands = self._run("existing-role")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(args[:3] == ["iam", "roles", "create"] for args in commands))

    def test_refuse_public_ssh(self):
        self._assert_no_grants("public-ssh")

    def test_refuse_broad_group(self):
        self._assert_no_grants("broad-group")

    def test_stop_on_read_failure(self):
        self._assert_no_grants("read-failure")

    def _assert_no_grants(self, scenario):
        result, commands = self._run(scenario)
        self.assertNotEqual(result.returncode, 0)
        for args in commands:
            self.assertNotIn("create", args)
            self.assertNotIn("add-iam-policy-binding", args)


if __name__ == "__main__":
    unittest.main()
