"""Safety rails shared by every Python test.

Three guarantees, enforced here so no single test has to remember them:

1. **A test cannot spend money.** Sockets leaving this machine are refused and the
   developer's cloud, Hugging Face and model-provider credentials are hidden, so a test
   can never reach a paid API, launch a cloud machine or upload anything. Stub ``sky``,
   ``gcloud``, ``kubectl``, ``aws`` and ``az`` commands sit first on ``PATH`` and fail.
2. **A test cannot hang the suite.** Each test has a deadline (60 s by default); when it
   expires the test fails with the stack of where it was stuck.
3. **A test sees the same machine everywhere.** With real credentials and cloud tools
   hidden, a laptop behaves like CI instead of probing whatever is installed on it.

Opt out per test only for a real reason::

    @pytest.mark.allow_network          # talks to a real remote service on purpose
    @pytest.mark.deadline(180)          # measured, legitimately slow

Environment: ``JENSEN_TEST_DEADLINE`` (seconds, ``0`` disables),
``JENSEN_TEST_IMPACT`` (write the test-impact map; see ``scripts/check.py``).
"""

import atexit
import faulthandler
import ipaddress
import json
import os
import shutil
import signal
import socket
import stat
import sys
import tempfile
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEADLINE = float(os.environ.get("JENSEN_TEST_DEADLINE", "60"))
SECRETS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_SECRET",
    "RUNPOD_API_KEY",
    "LAMBDA_API_KEY",
    "WANDB_API_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
)
CLOUD_COMMANDS = ("sky", "gcloud", "gsutil", "kubectl", "aws", "az")

# --- 1. No spending -------------------------------------------------------------------
# Applied at import, before any test module imports application code that reads these.
_sandbox = Path(tempfile.mkdtemp(prefix="jensen-test-isolation-"))
atexit.register(shutil.rmtree, _sandbox, ignore_errors=True)
for _name in SECRETS:
    os.environ.pop(_name, None)
os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HOME=str(_sandbox / "huggingface"),
    HF_TOKEN_PATH=str(_sandbox / "huggingface/token"),
    CLOUDSDK_CONFIG=str(_sandbox / "gcloud"),
    AWS_SHARED_CREDENTIALS_FILE=str(_sandbox / "aws-credentials"),
    AWS_CONFIG_FILE=str(_sandbox / "aws-config"),
    KUBECONFIG=str(_sandbox / "kubeconfig"),
    WANDB_MODE="disabled",
)
_blocked_commands = _sandbox / "blocked-commands.log"
if os.name == "posix":
    _bin = _sandbox / "bin"
    _bin.mkdir()
    for _name in CLOUD_COMMANDS:
        _stub = _bin / _name
        _stub.write_text(
            "#!/bin/sh\n"
            f'echo "{_name} $*" >> "{_blocked_commands}"\n'
            f'echo "tests/conftest.py blocks the real {_name} command" >&2\n'
            "exit 97\n"
        )
        _stub.chmod(_stub.stat().st_mode | stat.S_IXUSR)
    os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

_connect, _connect_ex = socket.socket.connect, socket.socket.connect_ex
_getaddrinfo = socket.getaddrinfo
_local_names = {"localhost", "testserver", "", None}
_state = threading.local()


def _is_local(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if host in _local_names or (isinstance(host, str) and host.endswith(".localhost")):
        return True
    try:
        address = ipaddress.ip_address(host.split("%")[0])
    except AttributeError, ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _refuse(target: str):
    attempts = getattr(_state, "attempts", None)
    if attempts is not None:
        attempts.append(target)
    # An ordinary "offline" error, so application code takes its real failure path.
    raise ConnectionRefusedError(f"tests/conftest.py blocks network access to {target}")


def _allowed() -> bool:
    return getattr(_state, "allow", False)


def _guard_connect(self, address, *args):
    if (
        self.family in (socket.AF_INET, socket.AF_INET6)
        and not _allowed()
        and not _is_local(address[0])
    ):
        _refuse(f"{address[0]}:{address[1]}")
    return _connect(self, address, *args)


def _guard_connect_ex(self, address, *args):
    if (
        self.family in (socket.AF_INET, socket.AF_INET6)
        and not _allowed()
        and not _is_local(address[0])
    ):
        _refuse(f"{address[0]}:{address[1]}")
    return _connect_ex(self, address, *args)


def _guard_getaddrinfo(host, *args, **kwargs):
    if not _allowed() and not _is_local(host):
        attempts = getattr(_state, "attempts", None)
        if attempts is not None:
            attempts.append(f"DNS lookup of {host}")
        raise socket.gaierror(socket.EAI_NONAME, f"tests/conftest.py blocks DNS for {host}")
    return _getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _guard_connect
socket.socket.connect_ex = _guard_connect_ex
socket.getaddrinfo = _guard_getaddrinfo


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "allow_network: this test reaches a real remote service on purpose"
    )
    config.addinivalue_line("markers", "deadline(seconds): replace the default per-test deadline")


@pytest.fixture(autouse=True)
def _no_spending(request):
    """Fail a test that tried to leave this machine, even if the application hid the error."""
    allow = request.node.get_closest_marker("allow_network") is not None
    _state.allow, _state.attempts = allow, []
    size = _blocked_commands.stat().st_size if _blocked_commands.exists() else 0
    yield
    attempts, _state.attempts, _state.allow = _state.attempts, None, False
    commands = []
    if _blocked_commands.exists() and _blocked_commands.stat().st_size > size:
        with _blocked_commands.open() as handle:
            handle.seek(size)
            commands = [line.strip() for line in handle if line.strip()]
    if allow:
        return
    problems = [f"network: {item}" for item in dict.fromkeys(attempts)]
    problems += [f"cloud command: {item}" for item in dict.fromkeys(commands)]
    if problems:
        pytest.fail(
            "This test tried to leave the machine. Tests must not reach real services "
            "(cost, flakiness, private data). Replace the call with a local fixture, or mark "
            "the test @pytest.mark.allow_network if it is a deliberate live check.\n  "
            + "\n  ".join(problems[:10]),
            pytrace=False,
        )


# --- 3. Same machine everywhere -------------------------------------------------------
_disk_usage = shutil.disk_usage


@pytest.fixture(autouse=True)
def _roomy_disk(monkeypatch):
    """Results must not depend on how full the developer's disk is.

    Many features keep 10 GiB of headroom. On a nearly full laptop that made unrelated
    tests fail, and one wait forever. Tests about low disk space patch this themselves.
    """

    def roomy(path):
        usage = _disk_usage(path)
        return usage._replace(free=max(usage.free, 1 << 40))

    monkeypatch.setattr(shutil, "disk_usage", roomy)


# --- 2. No hanging --------------------------------------------------------------------
class DeadlineExpired(BaseException):
    """Not an ``Exception``: application ``except Exception`` blocks must not swallow it."""


@pytest.fixture(autouse=True)
def _deadline(request):
    marker = request.node.get_closest_marker("deadline")
    limit = float(marker.args[0]) if marker and marker.args else DEADLINE
    usable = (
        limit > 0
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not usable:
        yield
        return

    def expire(signum, frame):
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        raise DeadlineExpired(
            f"{request.node.nodeid} ran longer than {limit:g} s. It is stuck, not slow: see "
            "the thread stacks above. Mark a measured slow test with @pytest.mark.deadline(N)."
        )

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except DeadlineExpired as exc:
        pytest.fail(str(exc), pytrace=False)


# --- 4. Test-impact map: which source files does each test file execute? ---------------
# Recorded only when JENSEN_TEST_IMPACT names an output directory. ``scripts/check.py``
# uses the result to run just the tests a change can affect.
_IMPACT = os.environ.get("JENSEN_TEST_IMPACT")
_SOURCES = (str(ROOT / "packages/core/src") + os.sep, str(ROOT / "scripts") + os.sep)
_seen: dict[str, set[str]] = {}
_current: set[str] | None = None

if _IMPACT:
    _TOOL = sys.monitoring.COVERAGE_ID
    sys.monitoring.use_tool_id(_TOOL, "jensen-test-impact")

    def _entered(code, offset):
        name = code.co_filename
        if _current is not None and name.startswith(_SOURCES):
            _current.add(name)
            return None  # Keep watching: another test file may enter the same function.
        return sys.monitoring.DISABLE if not name.startswith(_SOURCES) else None

    sys.monitoring.register_callback(_TOOL, sys.monitoring.events.PY_START, _entered)
    sys.monitoring.set_events(_TOOL, sys.monitoring.events.PY_START)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _current
    if not _IMPACT:
        return (yield)
    name = Path(str(item.fspath)).resolve().relative_to(ROOT).as_posix()
    _current = _seen.setdefault(name, set())
    try:
        return (yield)
    finally:
        _current = None


def pytest_sessionfinish(session):
    if not _IMPACT:
        return
    # Stop watching before interpreter shutdown starts clearing this module's globals.
    sys.monitoring.set_events(_TOOL, 0)
    sys.monitoring.register_callback(_TOOL, sys.monitoring.events.PY_START, None)
    if not _seen:
        return
    directory = Path(_IMPACT)
    directory.mkdir(parents=True, exist_ok=True)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    document = {
        test: sorted(Path(name).relative_to(ROOT).as_posix() for name in sources)
        for test, sources in _seen.items()
    }
    (directory / f"{worker}.json").write_text(json.dumps(document, sort_keys=True))
