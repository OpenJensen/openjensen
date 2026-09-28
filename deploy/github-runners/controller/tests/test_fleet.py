from datetime import UTC, datetime, timedelta
from enum import Enum
from unittest.mock import Mock

import pytest
from config import Settings
from fleet import Fleet, queued_jobs

REPO = "owner/repo"
SETTINGS = Settings("project", "zone", "image", "subnet", REPO, "app", "install", "bucket")
NAME = "firebird-ci-0-" + "a" * 32


def vm(age=0, status="RUNNING", slot="firebird-ci-0", name=NAME):
    return {
        "name": slot,
        "status": status,
        "creationTimestamp": (datetime.now(UTC) - timedelta(seconds=age)).isoformat(),
        "metadata": {"items": [{"key": "runner-name", "value": name}]},
    }


class Activity(Enum):
    IDLE = "idle"
    BUSY = "busy"


def runner(activity=Activity.IDLE, status="online", name=NAME):
    return {"id": 1, "name": name, "busy": activity == Activity.BUSY, "status": status}


def work(count, status="queued", labels=None):
    runs = [{"id": 1, "event": "push", "head_repository": {"full_name": REPO}}]
    jobs = {
        1: [
            {
                "id": index,
                "status": status,
                "runner_name": NAME,
                "labels": labels or ["self-hosted", "linux", "x64", "firebird-gcp"],
            }
            for index in range(count)
        ]
    }
    return runs, jobs


def fleet(count=0, vms=(), runners=()):
    github, compute = Mock(), Mock()
    github.runners.return_value = list(runners)
    github.work.return_value = work(count)
    github.retire.return_value = True
    github.register.return_value = {"encoded_jit_config": "secret"}
    compute.instances.return_value = list(vms)
    return Fleet(SETTINGS, github, compute), github, compute


@pytest.mark.parametrize("count,expected", [(0, 0), (1, 1), (3, 3), (4, 3), (100, 3)])
def test_three_worker_cap(count, expected):
    service, github, compute = fleet(count)
    service.reconcile()
    assert compute.create.call_count == expected
    assert github.register.call_count == expected
    assert {call.args[0] for call in compute.create.call_args_list} <= set(SETTINGS.slots())


def test_busy_slot_capacity():
    service, _, compute = fleet(4, [vm()], [runner(activity=Activity.BUSY)])
    service.reconcile()
    assert compute.create.call_count == 2
    compute.delete.assert_not_called()


def test_boot_reserves_slot():
    service, _, compute = fleet(1, [vm(status="PROVISIONING")])
    service.reconcile()
    compute.create.assert_not_called()
    compute.delete.assert_not_called()


def test_active_job_protection():
    service, github, compute = fleet(0, [vm(age=1000)], [runner()])
    github.work.return_value = work(1, status="in_progress")
    service.reconcile()
    github.retire.assert_not_called()
    compute.delete.assert_not_called()


def test_retire_before_delete():
    service, github, compute = fleet(0, [vm(age=200)], [runner()])
    order = Mock()
    order.attach_mock(github.retire, "retire")
    order.attach_mock(compute.delete, "delete")
    service.reconcile()
    assert [call[0] for call in order.mock_calls] == ["retire", "delete"]


def test_assignment_race():
    service, github, compute = fleet(0, [vm(age=200)], [runner()])
    github.retire.return_value = False
    service.reconcile()
    compute.delete.assert_not_called()


@pytest.mark.parametrize("status", ["TERMINATED", "RUNNING"])
def test_dead_worker_cleanup(status):
    service, _, compute = fleet(1, [vm(age=1000, status=status)])
    service.reconcile()
    compute.delete.assert_called_once_with("firebird-ci-0")
    # Deletion is asynchronous: do not reuse the slot within this reconciliation.
    assert all(call.args[0] != "firebird-ci-0" for call in compute.create.call_args_list)


def test_stopping_slot():
    service, _, compute = fleet(3, [vm(status="STOPPING")])
    service.reconcile()
    assert compute.create.call_count == 2
    compute.delete.assert_not_called()


def test_fail_closed_reads():
    service, github, compute = fleet(0, [vm(age=200)], [runner()])
    github.work.side_effect = RuntimeError("rate limited")
    with pytest.raises(RuntimeError):
        service.reconcile()
    github.retire.assert_not_called()
    compute.delete.assert_not_called()
    compute.create.assert_not_called()


def test_orphan_cleanup_is_scoped():
    runners = [
        runner(),
        runner(name="another-pool"),
        runner(name="firebird-ci-0-other"),
        runner(activity=Activity.BUSY, name="firebird-ci-1-" + "b" * 32),
    ]
    service, github, _ = fleet(runners=runners)
    service.reconcile()
    github.retire.assert_called_once_with(1)


def test_ambiguous_creation():
    service, github, compute = fleet(3)
    compute.create.side_effect = RuntimeError("ambiguous timeout")
    with pytest.raises(RuntimeError):
        service.reconcile()
    assert github.register.call_count == 1
    assert compute.create.call_count == 1


@pytest.mark.parametrize("labels", [["ubuntu-latest"], ["firebird-gcp", "gpu"], ["self-hosted"]])
def test_unrelated_labels(labels):
    runs, jobs = work(1, labels=labels)
    assert queued_jobs(runs, jobs, REPO) == 0


@pytest.mark.parametrize("event", ["pull_request_target", "workflow_run", "unknown"])
def test_untrusted_event(event):
    runs, jobs = work(1)
    runs[0]["event"] = event
    assert queued_jobs(runs, jobs, REPO) == 0


def test_fork_is_ignored():
    runs, jobs = work(1)
    runs[0]["head_repository"]["full_name"] = "fork/repo"
    assert queued_jobs(runs, jobs, REPO) == 0


def test_jobs_are_deduplicated():
    runs, jobs = work(1)
    jobs[1] *= 2
    assert queued_jobs(runs, jobs, REPO) == 1


def test_deleted_head_repo():
    runs, jobs = work(1)
    runs[0]["head_repository"] = None
    assert queued_jobs(runs, jobs, REPO) == 0
