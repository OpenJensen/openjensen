import importlib.util
import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location("monitor", Path(__file__).parents[1] / "monitor.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(job_id=9, run_id="isaac-act-test-ab12", experimental=True)

    def test_reject_wrong_or_missing_task_identity(self):
        rows = [
            dict(job_id=9, job_name=self.args.run_id, task_name=t, status="RUNNING")
            for t in monitor.TASKS
        ]
        self.assertEqual(monitor.select_tasks(rows, 9, self.args.run_id)["isaac"], "RUNNING")
        for bad in (
            rows[:1],
            rows + rows[:1],
            [{**r, "job_id": 8} for r in rows],
            [{**r, "job_name": "another-group"} for r in rows],
        ):
            with self.assertRaises(ValueError):
                monitor.select_tasks(bad, 9, self.args.run_id)

    def test_success_does_not_invent_quality_or_execution_evidence(self):
        def fetch(_, kind):
            return {"isaac": "SUCCEEDED", "vla": "CANCELLED"} if kind == "queue" else "real log"

        snapshot = monitor.collect(self.args, monitor.initial(self.args), fetch)
        self.assertEqual(snapshot["status"], "SUCCEEDED")
        self.assertIsNotNone(snapshot["collected_at"])
        self.assertIsNone(snapshot["collection_error"])
        self.assertIsNone(snapshot["outcomes"]["pickup_success"])
        self.assertIsNone(snapshot["outcomes"]["rollout_completed"])
        self.assertEqual(snapshot["outcomes"]["calibration"], "unverified")

    def test_timeout_retains_previous_evidence(self):
        previous = monitor.initial(self.args)
        previous.update(status="RUNNING", collected_at="2026-09-27T00:00:00+00:00")
        previous["logs"]["isaac"] = "last known"

        def fetch(_, kind):
            raise subprocess.TimeoutExpired("probe", 40)

        snapshot = monitor.collect(self.args, previous, fetch)
        self.assertEqual(snapshot["collected_at"], previous["collected_at"])
        self.assertEqual(snapshot["logs"], previous["logs"])
        self.assertTrue(snapshot["collection_error"])
        self.assertEqual(previous["collection_error"], "Waiting for the first remote collection.")

    def test_partial_log_failure_does_not_claim_freshness(self):
        previous = monitor.initial(self.args)

        def fetch(_, kind):
            if kind == "queue":
                return {"isaac": "RUNNING", "vla": "RUNNING"}
            if kind == "vla":
                raise ValueError("sensitive provider error must not leak")
            return "current isaac log"

        snapshot = monitor.collect(self.args, previous, fetch)
        self.assertIsNone(snapshot["collected_at"])
        self.assertIn("vla", snapshot["collection_error"])
        self.assertNotIn("sensitive", snapshot["collection_error"])
        self.assertEqual(previous["logs"]["isaac"], "")

    def test_chunked_redaction_and_utf8_bounds(self):
        tail = monitor.LogTail()
        for chunk in [
            "access_to",
            "ken=hidden\n",
            "-----BE",
            "GIN PRIVATE KEY-----\n",
            "private\n",
            "-----END PRIVATE KEY-----\n",
            "Bearer abc",
            ".def\n",
        ]:
            tail.write(chunk)
        self.assertNotIn("hidden", tail.value)
        self.assertNotIn("private\n", tail.value)
        self.assertNotIn("abc.def", tail.value)
        tail.write(("🦜" * 500 + "\n") * 100)
        self.assertLessEqual(len(tail.value.encode()), monitor.LOG_BYTES)
        tail.write("x" * (monitor.LOG_BYTES + 1))
        tail.write("more\n")
        self.assertIn("oversized log line omitted", tail.value)

    def test_supported_account_credentials_are_redacted_across_sdk_chunks(self):
        samples = [
            ("HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz\n", "hf_abcdefghijklmnopqrstuvwxyz"),
            ("Downloaded with hf_abc_def-12345678\n", "hf_abc_def-12345678"),
            ("SDK token ya29.abc_def-12345678\n", "ya29.abc_def-12345678"),
            ("Authorization: Basic cHJpdmF0ZTpwdw==\n", "cHJpdmF0ZTpwdw=="),
            ("Proxy-Authorization=Bearer proxy-private\n", "proxy-private"),
            ("Cookie: session=private-session; csrf=private-csrf\n", "private-session"),
            ("Set-Cookie: session=private-cookie; HttpOnly\n", "private-cookie"),
            ('{"client_secret": "private secret with spaces"}\n', "private secret with spaces"),
            ('{"Authorization": "Bearer private-json-auth"}\n', "private-json-auth"),
            ('password="unfinished private secret', "unfinished private secret"),
            ("https://user:private-url-password@example.com/file\n", "private-url-password"),
            ("https://example.com/file?token=private-query&part=1\n", "private-query"),
            ("https://example.com/file?access%5Ftoken=encoded-private\n", "encoded-private"),
            ("https://example.com/file?X-Goog-Signature=signed-private\n", "signed-private"),
            ("https://example.com/file#access_token=fragment-private\n", "fragment-private"),
        ]
        for text, private in samples:
            with self.subTest(text=text):
                tail = monitor.LogTail()
                for index in range(0, len(text), 3):
                    tail.write(text[index : index + 3])
                self.assertNotIn(private, tail.value)
                self.assertIn("[redacted]", tail.value)
                self.assertLessEqual(len(tail.value.encode()), monitor.LOG_BYTES)

    def test_redaction_preserves_nonsecret_training_details_and_url_parameters(self):
        normal = (
            "step=100 tokens=32 token_count=64 tokenizer=SmolVLM status=RUNNING\n"
            "https://huggingface.co/lerobot/smolvla_base?revision=abc&file=config.json\n"
        )
        self.assertEqual(monitor.clean_log(normal), normal)
        protected = monitor.clean_log("https://host/model?token=private&part=2&revision=abc")
        self.assertIn("part=2&revision=abc", protected)

    def test_http_200_log_stream_error_must_check_final_request_result(self):
        response, common, sdk, payloads = Mock(), Mock(), Mock(), Mock()
        common.make_authenticated_request.return_value = response
        common.get_request_id.return_value = "request-123"
        payloads.JobsLogsBody.return_value.model_dump_json.return_value = "{}"

        def streamed(**kwargs):
            kwargs["output_stream"].write("stream started normally\n")
            if kwargs["get_result"]:
                raise RuntimeError("server-side log request failed after HTTP 200")

        sdk.stream_response.side_effect = streamed
        with self.assertRaises(RuntimeError):
            monitor.read_logs(9, "isaac", monitor.LogTail(), common, sdk, payloads)
        payloads.JobsLogsBody.assert_called_once_with(
            job_id=9, task="isaac", follow=False, refresh=False, tail=200
        )
        response.close.assert_called_once()
        self.assertTrue(sdk.stream_response.call_args.kwargs["get_result"])

    def test_failed_job_exit_code_is_not_a_collection_failure(self):
        response, common, sdk, payloads = Mock(), Mock(), Mock(), Mock()
        common.make_authenticated_request.return_value = response
        payloads.JobsLogsBody.return_value.model_dump_json.return_value = "{}"
        sdk.stream_response.return_value = 100
        monitor.read_logs(9, "isaac", monitor.LogTail(), common, sdk, payloads)
        response.close.assert_called_once()

    def test_atomic_private_publish_and_symlink_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            feed = Path(directory) / "feed"
            snapshot = monitor.initial(self.args)
            monitor.publish(feed, snapshot)
            target = feed / (self.args.run_id + ".json")
            self.assertEqual(json.loads(target.read_text()), snapshot)
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(feed.iterdir()), [target])
            target.unlink()
            target.symlink_to(Path(directory) / "elsewhere")
            with self.assertRaises(ValueError):
                monitor.publish(feed, snapshot)

    def test_restart_preserves_evidence_and_rejects_mismatched_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            self.args.output_dir = Path(directory)
            previous = monitor.initial(self.args)
            previous.update(status="RUNNING", collected_at="2026-09-26T00:00:00+00:00")
            previous["logs"]["isaac"] = "last known evidence"
            monitor.publish(self.args.output_dir, previous)
            restored = monitor.resume(self.args)
            self.assertEqual(restored["collected_at"], previous["collected_at"])
            self.assertEqual(restored["logs"], previous["logs"])
            self.assertTrue(restored["collection_error"])
            previous["job_id"] = "999"
            monitor.publish(self.args.output_dir, previous)
            with self.assertRaises(ValueError):
                monitor.resume(self.args)

    def test_restart_rejects_oversized_logs_and_invalid_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            self.args.output_dir = Path(directory)
            for changes in (
                {"logs": {"isaac": "x" * (monitor.LOG_BYTES + 1), "vla": ""}},
                {
                    "outcomes": {
                        "rollout_completed": 1,
                        "pickup_success": None,
                        "calibration": "unverified",
                    }
                },
            ):
                snapshot = {**monitor.initial(self.args), **changes}
                monitor.publish(self.args.output_dir, snapshot)
                with self.assertRaises(ValueError):
                    monitor.resume(self.args)


if __name__ == "__main__":
    unittest.main()
