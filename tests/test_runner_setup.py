"""Package maintenance must not stop a disposable runner during its job."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/prepare_runner.sh"
GUARD = "runner.environment == 'self-hosted' && runner.os == 'Linux'"


def test_gcp_reuses_browser_dependencies():
    workflow = yaml.safe_load((ROOT / ".github/workflows/application.yml").read_text())
    installers = [
        step
        for step in workflow["jobs"]["web"]["steps"]
        if "playwright install" in step.get("run", "")
    ]
    # Three installers since the hosted lane grew a browser cache:
    # 1. GCP: browsers ship in the image, runner script only registers.
    # 2. Hosted cold cache: browsers + OS deps.
    # 3. Hosted cache hit: OS deps only (browser binaries restored from cache).
    assert len(installers) == 3
    by_name = {step["name"]: step for step in installers}
    gcp = by_name["Use preinstalled GCP browser dependencies"]
    assert "--with-deps" not in gcp["run"]
    assert gcp["if"] == "runner.environment == 'self-hosted' && runner.os == 'Linux'"
    cold = by_name["Install hosted browser and OS dependencies (cold cache)"]
    assert "--with-deps" in cold["run"]
    assert "cache-hit != 'true'" in cold["if"]
    assert "runner.environment != 'self-hosted'" in cold["if"]
    deps_only = by_name["Install hosted OS dependencies only (browser restored from cache)"]
    assert "--with-deps" not in deps_only["run"]
    assert "install-deps" in deps_only["run"]
    assert "cache-hit == 'true'" in deps_only["if"]
    video = next(
        step
        for step in workflow["jobs"]["web"]["steps"]
        if step.get("name") == "Install video validation tools (Linux)"
    )
    assert video["if"] == "runner.os == 'Linux' && runner.environment != 'self-hosted'"


@pytest.mark.skipif(os.name != "posix", reason="Linux runner setup requires a POSIX shell")
def test_runner_upgrade_policy(tmp_path):
    config = tmp_path / "etc"
    commands = tmp_path / "commands"
    systemctl = tmp_path / "systemctl"
    # Inspect configuration when timers are masked; protection must already exist.
    systemctl.write_text(
        "#!/bin/bash\n"
        'test -s "$RUNNER_CONFIG_ROOT/needrestart/conf.d/firebird-ci.conf" || exit 1\n'
        'test -s "$RUNNER_CONFIG_ROOT/apt/apt.conf.d/99firebird-ci" || exit 1\n'
        'printf "%s\\n" "$*" >> "$COMMAND_LOG"\n'
    )
    systemctl.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "RUNNER_CONFIG_ROOT": str(config),
        "COMMAND_LOG": str(commands),
    }
    for _ in range(2):
        subprocess.run(["bash", str(SCRIPT)], env=env, check=True)

    assert (
        commands.read_text().splitlines()
        == [
            "mask --now apt-daily.timer apt-daily-upgrade.timer",
        ]
        * 2
    )
    policy = config / "apt/apt.conf.d/99firebird-ci"
    assert 'APT::Periodic::Enable "0";' in policy.read_text()
    # Execute needrestart's Perl configuration, checking only our service is exempt.
    subprocess.run(
        [
            "perl",
            "-e",
            "our %nrconf = (override_rc => {}); my $loaded = do $ARGV[0]; "
            "die $@ || $! unless defined $loaded; "
            "my @rules = keys %{$nrconf{override_rc}}; "
            'die unless @rules == 1 && "firebird-runner.service" =~ $rules[0] '
            '&& "ssh.service" !~ $rules[0] && $nrconf{override_rc}{$rules[0]} == 0;',
            str(config / "needrestart/conf.d/firebird-ci.conf"),
        ],
        check=True,
    )


@pytest.mark.parametrize("workflow", ["application.yml", "simulation.yml"])
def test_jobs_prepare_before_setup(workflow):
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    for job in document["jobs"].values():
        steps = job.get("steps", [])
        for index, step in enumerate(steps):
            if not step.get("uses", "").startswith("actions/checkout@"):
                continue
            preparation = steps[index + 1]
            assert preparation.get("if") == GUARD
            assert preparation.get("working-directory") == "${{ github.workspace }}"
            assert preparation.get("run") == "sudo bash .github/scripts/prepare_runner.sh"
