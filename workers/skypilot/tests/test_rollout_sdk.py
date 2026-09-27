"""Offline submission identity and final-state regressions; no cloud calls."""

import importlib.util
import inspect
import io
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import rollout_sdk as adapter  # noqa: E402


def receipt():
    return {
        "schema_version": 1,
        "skypilot": "0.13.0",
        "job_id": 7,
        "group_name": "isaac-test-unique",
        "request_id": "launch-request-1",
        "tasks": [
            {"task_id": 0, "task_name": "isaac", "is_primary_in_job_group": True},
            {"task_id": 1, "task_name": "vla", "is_primary_in_job_group": False},
        ],
        "attached_exit_code": 101,
    }


def rows(primary="SUCCEEDED", auxiliary="CANCELLED"):
    submitted = receipt()
    return [
        {
            **task,
            "job_id": submitted["job_id"],
            "job_name": submitted["group_name"],
            "execution": "parallel",
            "is_job_group": True,
            "status": state,
        }
        for task, state in zip(submitted["tasks"], (primary, auxiliary), strict=True)
    ]


def dag():
    return SimpleNamespace(
        name="isaac-test-unique",
        primary_tasks=["isaac"],
        is_job_group=lambda: True,
        tasks=[SimpleNamespace(name="isaac"), SimpleNamespace(name="vla")],
    )


def sdk():
    sky = Mock()
    sky.jobs.launch.return_value = "launch-request-1"
    sky.stream_and_get.return_value = ([7], None)
    sky.jobs.tail_logs.return_value = 101
    sky.jobs.queue_v2.return_value = "queue-request-1"
    sky.get.return_value = (rows(), 1, {"SUCCEEDED": 1}, 1)
    return sky


class SDKTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.receipt_path = self.root / "receipt.json"
        self.task = self.root / "task.yaml"
        self.task.write_text("trusted prepared YAML fixture\n")
        self.sky = sdk()

    def launch(self, **kwargs):
        with patch.object(adapter, "load_dag", return_value=dag()):
            return adapter.launch(self.sky, self.task, self.receipt_path, **kwargs)

    def test_launcher_reconciles_recorded_101_instead_of_returning_raw_cli_exit(self):
        spec = importlib.util.spec_from_file_location(
            "status_test_launcher", _ROOT / "rollout_launch.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)

        def dispatch(command, **kwargs):
            if "rollout-sdk" not in command:
                return SimpleNamespace(returncode=101)
            target = Path(command[command.index("--receipt") + 1])
            result = adapter.launch(self.sky, self.task, target)
            return SimpleNamespace(returncode=result)

        with (
            patch.object(launcher, "_run_sdk", side_effect=dispatch),
            patch.object(adapter, "load_dag", return_value=dag()),
            patch.object(adapter, "collect", return_value=rows()),
            patch.object(launcher.tempfile, "mkdtemp", return_value=str(self.root)),
        ):
            self.assertEqual(
                launcher._launch(
                    [{"name": "isaac-test-unique"}],
                    self.task,
                    launcher._Mode.EXPERIMENTAL,
                ),
                0,
            )

    def test_recorded_transition_preserves_101_and_binds_exact_submission(self):
        with (
            patch.object(
                adapter, "collect", side_effect=[rows(auxiliary="CANCELLING"), rows()]
            ) as collect,
            patch.object(adapter.time, "sleep"),
        ):
            self.assertEqual(self.launch(), 0)
        result = adapter.read_json(self.receipt_path)
        self.assertEqual(result["job_id"], 7)
        self.assertEqual(result["attached_exit_code"], 101)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["phase"], "reconciled")
        self.assertEqual(result["first_observation"]["tasks"]["vla"], "CANCELLING")
        self.assertEqual(
            result["last_observation"],
            {
                "tasks": {"isaac": "SUCCEEDED", "vla": "CANCELLED"},
                "derived_group_status": "SUCCEEDED",
            },
        )
        self.assertEqual(collect.call_count, 2)
        self.sky.jobs.launch.assert_called_once()
        self.sky.jobs.tail_logs.assert_called_once_with(
            name=None,
            job_id=7,
            follow=True,
            controller=False,
            refresh=False,
        )

    def test_receipt_is_persisted_before_waiting_and_following(self):
        def wait(request):
            saved = adapter.read_json(self.receipt_path)
            self.assertEqual(saved["request_id"], request)
            self.assertEqual(saved["phase"], "submitted_request")
            return ([7], None)

        def tail(**kwargs):
            saved = adapter.read_json(self.receipt_path)
            self.assertEqual(saved["job_id"], kwargs["job_id"])
            self.assertEqual(saved["phase"], "submitted")
            return 0

        self.sky.stream_and_get.side_effect = wait
        self.sky.jobs.tail_logs.side_effect = tail
        self.assertEqual(self.launch(), 0)

    def test_yes_and_detached_preserve_submission_semantics(self):
        for yes in (False, True):
            with self.subTest(yes=yes):
                self.receipt_path.unlink(missing_ok=True)
                self.sky.reset_mock()
                with patch.object(adapter, "collect") as collect:
                    self.assertEqual(self.launch(yes=yes, detached=True), 0)
                self.assertIs(self.sky.jobs.launch.call_args.kwargs["_need_confirmation"], not yes)
                self.sky.jobs.tail_logs.assert_not_called()
                collect.assert_not_called()
                result = adapter.read_json(self.receipt_path)
                self.assertEqual(result["phase"], "submitted")
                self.assertNotIn("exit_code", result)

    def test_declined_confirmation_never_waits_tails_or_reconciles(self):
        self.sky.jobs.launch.side_effect = RuntimeError("confirmation declined")
        with self.assertRaisesRegex(RuntimeError, "declined"):
            self.launch()
        self.sky.jobs.launch.assert_called_once()
        self.sky.stream_and_get.assert_not_called()
        self.sky.jobs.tail_logs.assert_not_called()
        self.assertEqual(adapter.read_json(self.receipt_path)["phase"], "prepared")

    def test_request_id_missing_cannot_select_latest_request(self):
        for value in (None, "", "\n", True, 7, "a" * 129):
            with self.subTest(value=value):
                self.receipt_path.unlink(missing_ok=True)
                self.sky.jobs.launch.return_value = value
                with self.assertRaisesRegex(ValueError, "request ID"):
                    self.launch()
                self.sky.stream_and_get.assert_not_called()
                self.sky.jobs.tail_logs.assert_not_called()

    def test_invalid_job_ids_never_attach_or_relaunch(self):
        for value in (None, [], [1, 2], [True], True, [0], [-1], ["7"], "7"):
            with self.subTest(value=value):
                self.receipt_path.unlink(missing_ok=True)
                self.sky.reset_mock()
                self.sky.stream_and_get.return_value = (value, None)
                with self.assertRaisesRegex(ValueError, "job ID"):
                    self.launch()
                self.sky.jobs.launch.assert_called_once()
                self.sky.jobs.tail_logs.assert_not_called()
                self.assertEqual(
                    adapter.read_json(self.receipt_path)["request_id"], "launch-request-1"
                )

    def test_legacy_single_integer_job_id(self):
        self.sky.stream_and_get.return_value = (7, None)
        self.assertEqual(self.launch(detached=True), 0)
        self.assertEqual(adapter.read_json(self.receipt_path)["job_id"], 7)

    def test_lost_launch_response_is_not_retried(self):
        self.sky.stream_and_get.side_effect = OSError("lost response")
        with self.assertRaises(OSError):
            self.launch()
        self.sky.jobs.launch.assert_called_once()
        self.sky.jobs.tail_logs.assert_not_called()
        self.assertEqual(adapter.read_json(self.receipt_path)["request_id"], "launch-request-1")

    def test_receipt_must_be_writable_before_submission(self):
        self.receipt_path = self.root / "missing" / "receipt.json"
        with self.assertRaises(OSError):
            self.launch()
        self.sky.jobs.launch.assert_not_called()

    def test_existing_receipt_is_not_overwritten_or_resubmitted(self):
        self.receipt_path.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.launch()
        self.sky.jobs.launch.assert_not_called()
        self.assertEqual(self.receipt_path.read_text(), "preserve")

    def test_post_submission_receipt_failure_does_not_continue(self):
        write = adapter.write_json

        def fail_second(path, value):
            if value["phase"] == "submitted_request":
                raise OSError("disk full")
            write(path, value)

        output = io.StringIO()
        with (
            patch.object(adapter, "write_json", side_effect=fail_second),
            patch("sys.stdout", output),
            self.assertRaises(OSError),
        ):
            self.launch()
        self.assertIn("Launch request ID: launch-request-1", output.getvalue())
        self.sky.jobs.launch.assert_called_once()
        self.sky.stream_and_get.assert_not_called()
        self.sky.jobs.tail_logs.assert_not_called()

    def test_learned_job_id_is_printed_even_if_receipt_update_fails(self):
        write = adapter.write_json

        def fail_job_update(path, value):
            if value["phase"] == "submitted":
                raise OSError("disk full")
            write(path, value)

        output = io.StringIO()
        with (
            patch.object(adapter, "write_json", side_effect=fail_job_update),
            patch("sys.stdout", output),
            self.assertRaises(OSError),
        ):
            self.launch()
        self.assertIn("Submitted job 7 (isaac-test-unique).", output.getvalue())
        self.sky.jobs.tail_logs.assert_not_called()

    def test_invalid_dag_identity_prevents_submission(self):
        for key, value in (
            ("name", None),
            ("primary_tasks", None),
            ("primary_tasks", ["vla"]),
            ("tasks", [SimpleNamespace(name="isaac")]),
        ):
            with self.subTest(key=key):
                source = dag()
                setattr(source, key, value)
                with (
                    patch.object(adapter, "load_dag", return_value=source),
                    self.assertRaises(ValueError),
                ):
                    adapter.launch(self.sky, self.task, self.receipt_path)
                self.sky.jobs.launch.assert_not_called()

    def test_symlink_interchange_is_not_read_or_overwritten(self):
        target = self.root / "untouched.json"
        target.write_text("{}")
        self.receipt_path.symlink_to(target)
        with self.assertRaises(OSError):
            adapter.read_json(self.receipt_path)
        with self.assertRaises(ValueError):
            adapter.write_json(self.receipt_path, {"changed": True})
        self.assertEqual(target.read_text(), "{}")

    def test_other_exit_codes_never_reconcile(self):
        for code in (0, 1, 100, 102, 103, 130):
            with self.subTest(code=code):
                self.receipt_path.unlink(missing_ok=True)
                self.sky.jobs.tail_logs.return_value = code
                with patch.object(adapter, "collect") as collect:
                    self.assertEqual(self.launch(), code)
                collect.assert_not_called()
                self.assertEqual(adapter.read_json(self.receipt_path)["attached_exit_code"], code)

    def test_invalid_tail_code_fails_without_reconciliation(self):
        for code in (None, True, "101", -1, 256):
            with self.subTest(code=code):
                self.receipt_path.unlink(missing_ok=True)
                self.sky.jobs.tail_logs.return_value = code
                with patch.object(adapter, "collect") as collect, self.assertRaises(ValueError):
                    self.launch()
                collect.assert_not_called()

    def test_failed_cancelled_unknown_and_mismatched_tasks_cannot_be_success(self):
        invalid = [
            rows(primary="FAILED"),
            rows(primary="CANCELLED"),
            rows(auxiliary="FAILED_SETUP"),
            rows(primary="RUNNING", auxiliary="CANCELLED"),
            rows(auxiliary="NEW_STATE"),
            [],
            rows()[:1],
        ]
        for key, value in (
            ("job_id", 8),
            ("job_id", True),
            ("job_name", "different"),
            ("task_id", 1),
            ("task_id", False),
            ("is_primary_in_job_group", None),
            ("is_primary_in_job_group", False),
            ("is_primary_in_job_group", 1),
            ("is_job_group", 1),
            ("is_job_group", None),
            ("execution", "serial"),
        ):
            changed = rows()
            changed[0][key] = value
            invalid.append(changed)
        invalid.append([rows()[0], rows()[0]])
        for records in invalid:
            with self.subTest(records=records):
                self.receipt_path.unlink(missing_ok=True)
                with patch.object(adapter, "collect", return_value=records) as collect:
                    self.assertEqual(self.launch(), 101)
                self.assertEqual(collect.call_count, 1)
                self.assertEqual(adapter.read_json(self.receipt_path)["phase"], "unresolved")

    def test_auxiliary_success_is_an_allowed_terminal_state(self):
        with patch.object(adapter, "collect", return_value=rows(auxiliary="SUCCEEDED")):
            self.assertEqual(self.launch(), 0)

    def test_collection_errors_keep_original_101(self):
        for error in (
            OSError("unavailable"),
            ValueError("bad JSON"),
            subprocess.TimeoutExpired("observe", 40),
            subprocess.CalledProcessError(1, "observe"),
        ):
            with self.subTest(error=type(error).__name__):
                self.receipt_path.unlink(missing_ok=True)
                with patch.object(adapter, "collect", side_effect=error):
                    self.assertEqual(self.launch(), 101)
                self.assertEqual(adapter.read_json(self.receipt_path)["attached_exit_code"], 101)

    def test_deadline_caps_each_query_and_sleep_and_refuses_late_success(self):
        now = [0.0]
        timeouts = []

        def collect(path, timeout):
            timeouts.append(timeout)
            now[0] += 40 if len(timeouts) < 3 else timeout
            return rows(auxiliary="CANCELLING") if len(timeouts) < 3 else rows()

        with (
            patch.object(adapter, "collect", side_effect=collect),
            patch.object(adapter.time, "monotonic", side_effect=lambda: now[0]),
            patch.object(
                adapter.time, "sleep", side_effect=lambda delay: now.__setitem__(0, now[0] + delay)
            ),
        ):
            self.assertEqual(self.launch(), 101)
        self.assertEqual(timeouts, [40, 40, 36])
        self.assertEqual(now[0], 120)

    def test_read_only_query_has_exact_identity_and_no_refresh_or_latest_fallback(self):
        result = adapter.observe(self.sky, receipt())
        self.assertEqual(result, rows())
        self.sky.jobs.queue_v2.assert_called_once_with(
            refresh=False,
            skip_finished=False,
            all_users=False,
            job_ids=[7],
            limit=1,
            fields=list(adapter.QUEUE_FIELDS),
        )
        self.sky.get.assert_called_once_with("queue-request-1")
        self.sky.jobs.launch.assert_not_called()

    def test_typed_statuses_are_preserved_and_missing_queue_request_never_uses_latest(self):
        class State(Enum):
            SUCCEEDED = "SUCCEEDED"

        records = rows()
        records[0]["status"] = State.SUCCEEDED
        self.sky.get.return_value = (records, 1, {}, 1)
        self.assertEqual(adapter.observe(self.sky, receipt()), rows())
        self.sky.reset_mock()
        self.sky.jobs.queue_v2.return_value = None
        with self.assertRaisesRegex(ValueError, "request ID"):
            adapter.observe(self.sky, receipt())
        self.sky.get.assert_not_called()

    def test_invalid_receipt_cannot_issue_a_query(self):
        for key, value in (
            ("job_id", True),
            ("job_id", 0),
            ("schema_version", True),
            ("request_id", None),
            ("tasks", []),
            ("group_name", ""),
        ):
            with self.subTest(key=key):
                bad = receipt() | {key: value}
                with self.assertRaises(ValueError):
                    adapter.observe(self.sky, bad)
                self.sky.jobs.queue_v2.assert_not_called()

    def test_malformed_or_oversized_subprocess_output_is_rejected(self):
        for data in (b"{", b'{"job_id":1,"job_id":2}', b" " * (adapter.JSON_LIMIT + 1)):
            with self.subTest(length=len(data)):
                self.receipt_path.write_bytes(data)
                with self.assertRaises(ValueError):
                    adapter.read_json(self.receipt_path)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX launcher")
    def test_fifo_receipt_is_rejected_without_blocking(self):
        fifo = self.root / "status.pipe"
        os.mkfifo(fifo)
        probe = (
            "from pathlib import Path; import rollout_sdk; "
            "rollout_sdk.read_json(Path(__import__('sys').argv[1]))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe, str(fifo)],
            env=os.environ | {"PYTHONPATH": str(_ROOT)},
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid or oversized", result.stderr)

    def test_signal_between_child_creation_and_handle_return_cannot_orphan_child(self):
        original_popen = subprocess.Popen
        for number in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=number):
                created = []

                def interrupted_spawn(*args, **kwargs):
                    process = original_popen(*args, **kwargs)
                    created.append(process)
                    signal.raise_signal(number)
                    return process

                saved = signal.getsignal(number)
                try:
                    with patch.object(adapter.subprocess, "Popen", side_effect=interrupted_spawn):
                        with self.assertRaises((SystemExit, KeyboardInterrupt)):
                            adapter.run_owned(
                                [sys.executable, "-c", "import time; time.sleep(60)"],
                                timeout=5,
                            )
                    self.assertEqual(signal.getsignal(number), saved)
                    self.assertEqual(len(created), 1)
                    self.assertIsNotNone(created[0].poll())
                    with self.assertRaises(ProcessLookupError):
                        os.kill(created[0].pid, 0)
                finally:
                    for process in created:
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=3)

    def test_repeated_signal_during_cleanup_does_not_interrupt_kill_and_reap(self):
        original_popen = subprocess.Popen
        for number in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=number):
                created = []

                def spawn(*args, **kwargs):
                    process = original_popen(*args, **kwargs)
                    created.append(process)
                    original_wait = process.wait

                    def wait(timeout=None):
                        if timeout == 0.02:
                            signal.raise_signal(number)
                        return original_wait(timeout=timeout)

                    process.wait = wait
                    process.terminate = lambda: None  # Force the bounded kill fallback.
                    return process

                handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
                try:
                    with patch.object(adapter.subprocess, "Popen", side_effect=spawn):
                        with self.assertRaises(subprocess.TimeoutExpired):
                            adapter.run_owned(
                                [sys.executable, "-c", "import time; time.sleep(60)"],
                                timeout=0.01,
                                terminate_grace=0.02,
                            )
                    self.assertIsNotNone(created[0].poll())
                    for sig, handler in handlers.items():
                        self.assertEqual(signal.getsignal(sig), handler)
                finally:
                    for process in created:
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=3)

    def test_outer_wall_timeout_reaps_owned_hung_query_child(self):
        spec = importlib.util.spec_from_file_location(
            "timeout_launcher", _ROOT / "rollout_launch.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        child_pid = self.root / "child.pid"
        slow = self.root / "query.py"
        slow.write_text(
            "import os, pathlib, time\n"
            f"pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(60)\n"
        )
        supervisor = self.root / "supervisor.py"
        supervisor.write_text(
            "import signal, rollout_sdk\nfrom pathlib import Path\n"
            "signal.signal(signal.SIGTERM, rollout_sdk._terminated)\n"
            f"rollout_sdk.__file__ = {str(slow)!r}\n"
            "rollout_sdk.collect(Path('unused-receipt'), 40)\n"
        )
        # Wait until the owned query exists, then exercise the launcher's actual
        # wall-timeout cleanup. A separate unrelated process must remain alive.
        with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) as unrelated:
            try:
                with patch.dict(os.environ, {"PYTHONPATH": str(_ROOT)}):
                    with self.assertRaises(subprocess.TimeoutExpired):
                        launcher._run_sdk([sys.executable, str(supervisor)], timeout=1)
                self.assertTrue(child_pid.exists(), "query must start before testing outer timeout")
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(child_pid.read_text()), 0)
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=3)

    def test_real_hung_local_status_subprocess_is_bounded(self):
        # Exercise the actual production subprocess timeout without importing SkyPilot.
        slow = self.root / "slow.py"
        slow.write_text("import time\ntime.sleep(60)\n")
        original = adapter.__file__
        started = time.monotonic()
        with (
            patch.object(adapter, "__file__", str(slow)),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            adapter.collect(self.receipt_path, 0.15)
        self.assertEqual(adapter.__file__, original)
        self.assertLess(time.monotonic() - started, 5)


@unittest.skipUnless(
    importlib.util.find_spec("sky") or os.environ.get("SIM_REQUIRE_SKYPILOT") == "1",
    "Pinned SkyPilot SDK not installed (set SIM_REQUIRE_SKYPILOT=1 to require it)",
)
class PinnedLoaderCompatibilityTests(unittest.TestCase):
    def test_real_queue_record_types_and_missing_primary_metadata(self):
        adapter._runtime()
        from sky.jobs.state import ManagedJobStatus
        from sky.schemas.api.responses import ManagedJobRecord

        sky = sdk()
        records = [ManagedJobRecord(**row) for row in rows()]
        self.assertIs(records[0].is_job_group, True)
        self.assertIs(records[1].is_primary_in_job_group, False)
        self.assertEqual(records[0].execution, "parallel")
        self.assertIs(records[0].status, ManagedJobStatus.SUCCEEDED)
        sky.get.return_value = (records, 1, {}, 1)
        self.assertEqual(adapter.observe(sky, receipt()), rows())
        missing = rows()
        missing[1].pop("is_primary_in_job_group")
        sky.get.return_value = ([ManagedJobRecord(**row) for row in missing], 1, {}, 1)
        with self.assertRaisesRegex(ValueError, "primary role"):
            adapter.observe(sky, receipt())

    def test_real_sdk_declined_confirmation_matches_click_cli_exit_without_submission(self):
        # Exercise the pinned launch function/Click prompt but bypass server and
        # usage decorators. Every optimization/API operation is a local double.
        sky = adapter._runtime()
        import click
        from click.testing import CliRunner
        from sky.jobs.client import sdk as upstream

        actual_launch = inspect.unwrap(sky.jobs.launch)
        prepared = dag()

        @click.command()
        def original_cli():
            actual_launch(prepared, name=prepared.name, _need_confirmation=True)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            path.write_text("fixture")
            output = Path(directory) / "receipt.json"
            fake = sdk()
            fake.jobs.launch.side_effect = actual_launch

            @click.command()
            def adapter_cli():
                raise SystemExit(adapter.main())

            with (
                patch.object(upstream.versions, "get_remote_api_version", return_value=64),
                patch.object(
                    upstream.dag_utils, "convert_entrypoint_to_dag", return_value=prepared
                ),
                patch.object(
                    upstream.admin_policy_utils,
                    "apply_and_use_config_in_current_request",
                    side_effect=lambda *args, **kwargs: nullcontext(prepared),
                ),
                patch.object(upstream.sdk, "validate"),
                patch.object(upstream.sdk, "optimize", return_value="optimize-request") as optimize,
                patch.object(upstream.sdk, "stream_and_get") as stream,
                patch.object(upstream.server_common, "make_authenticated_request") as submit,
                patch("socket.socket.connect", side_effect=AssertionError("network forbidden")),
                patch("socket.getaddrinfo", side_effect=AssertionError("network forbidden")),
                patch.object(adapter, "_runtime", return_value=fake),
                patch.object(adapter, "load_dag", return_value=prepared),
                patch.object(
                    sys, "argv", ["adapter", "launch", str(path), "--receipt", str(output)]
                ),
            ):
                original = CliRunner().invoke(original_cli, input="n\n")
                self.assertEqual(original.exit_code, 1, original.output)
                self.assertIn("Aborted", original.output)
                repaired = CliRunner().invoke(adapter_cli, input="n\n")
                self.assertEqual(repaired.exit_code, original.exit_code, repaired.output)
                self.assertIn("Abort", repaired.output)
                self.assertEqual(optimize.call_count, 2)
                self.assertEqual(stream.call_count, 2)
                stream.assert_called_with("optimize-request")
                submit.assert_not_called()
                fake.stream_and_get.assert_not_called()
                fake.jobs.tail_logs.assert_not_called()
                fake.jobs.queue_v2.assert_not_called()
                self.assertEqual(adapter.read_json(output)["phase"], "prepared")

    def test_real_loader_keeps_parallel_primary_auxiliary_and_cli_defaults_offline(self):
        # No API/network credentials required: only the actual pinned parser is called.
        sky = adapter._runtime()
        from sky.jobs import utils as job_utils
        from sky.jobs.state import ManagedJobStatus
        from sky.utils import dag_utils

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "group.yaml"
            path.write_text(
                "name: isaac-test-unique\nexecution: parallel\nprimary_tasks: [isaac]\n"
                "termination_delay: 30s\n---\nname: isaac\nrun: 'true'\nresources:\n"
                "  job_recovery: {max_restarts_on_errors: 0}\n---\nname: vla\nrun: 'true'\n"
                "resources:\n  job_recovery: {max_restarts_on_errors: 0}\n"
            )
            with (
                patch.object(sky.jobs, "launch", side_effect=AssertionError("network forbidden")),
                patch.object(sky.jobs, "queue_v2", side_effect=AssertionError("network forbidden")),
                patch("socket.socket.connect", side_effect=AssertionError("network forbidden")),
                patch("socket.getaddrinfo", side_effect=AssertionError("network forbidden")),
            ):
                loaded = adapter.load_dag(path)
                self.assertEqual(
                    adapter.identity(loaded),
                    {key: receipt()[key] for key in ("tasks", "group_name")},
                )
                self.assertEqual(loaded.get_termination_delay_secs("vla"), 30)
                for task in loaded.tasks:
                    self.assertEqual(
                        next(iter(task.resources)).job_recovery["max_restarts_on_errors"], 0
                    )
                upstream = job_utils._get_job_status_from_tasks(rows())
                self.assertEqual(upstream, (ManagedJobStatus.SUCCEEDED, 0))
                self.assertTrue(callable(dag_utils.fill_default_config_in_dag_for_job_launch))


if __name__ == "__main__":
    unittest.main()
