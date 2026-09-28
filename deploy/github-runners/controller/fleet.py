"""Reconcile domain snapshots; cloud and GitHub I/O live in adapters."""

import re
import uuid
from datetime import UTC, datetime

from config import BOOT_GRACE_SECONDS, IDLE_GRACE_SECONDS, POOL_LABEL, RUNNER_LABELS

TERMINATED = "TERMINATED"
QUEUED = "queued"
ACTIVE_JOB = "in_progress"
ALLOWED_EVENTS = {"push", "pull_request", "workflow_dispatch", "merge_group"}


def queued_jobs(runs, jobs, repository):
    """Only jobs whose entire label requirement fits this pool count as demand."""
    supported = {label.lower() for label in RUNNER_LABELS}
    selected = set()
    for run in runs:
        if run["event"] not in ALLOWED_EVENTS:
            continue
        if (run.get("head_repository") or {}).get("full_name") != repository:
            continue
        for job in jobs[run["id"]]:
            labels = {label.lower() for label in job.get("labels", [])}
            if job["status"] != QUEUED or POOL_LABEL not in labels:
                continue
            if labels <= supported:
                selected.add(job["id"])
    return len(selected)


def _metadata(vm):
    return {item["key"]: item["value"] for item in vm.get("metadata", {}).get("items", [])}


def _age(vm, now):
    created = datetime.fromisoformat(vm["creationTimestamp"].replace("Z", "+00:00"))
    return (now - created).total_seconds()


class Fleet:
    def __init__(self, settings, github, compute):
        self._settings = settings
        self._github = github
        self._compute = compute

    def reconcile(self):
        # Finish every read before mutation: incomplete API snapshots never mean idle.
        slots = self._settings.slots()
        vms = self._compute.instances()
        runners = self._github.runners()
        runs, jobs = self._github.work()
        demand = queued_jobs(runs, jobs, self._settings.repository)
        now = datetime.now(UTC)
        running = {
            job.get("runner_name")
            for group in jobs.values()
            for job in group
            if job["status"] == ACTIVE_JOB
        }
        by_name = {runner["name"]: runner for runner in runners}
        occupied = {vm["name"] for vm in vms}
        available = 0

        for vm in vms:
            name = _metadata(vm).get("runner-name", "")
            runner = by_name.get(name)
            busy = name in running or (runner and runner["busy"])
            if busy and vm["status"] != TERMINATED:
                continue
            if vm["status"] == TERMINATED:
                self._compute.delete(vm["name"])
                continue
            if vm["status"] in {"STOPPING", "SUSPENDING", "SUSPENDED"}:
                continue
            age = _age(vm, now)
            stale = age > BOOT_GRACE_SECONDS and (not runner or runner["status"] == "offline")
            idle = demand == 0 and age > IDLE_GRACE_SECONDS
            if stale or idle:
                # GitHub rejects removal if assignment raced with this snapshot.
                if runner and not self._github.retire(runner["id"]):
                    continue
                self._compute.delete(vm["name"])
                continue
            available += 1

        # Remove only our orphan registrations, never other repository runners.
        live_names = {_metadata(vm).get("runner-name") for vm in vms}
        slot_pattern = "|".join(re.escape(slot) for slot in slots)
        pattern = rf"(?:{slot_pattern})-[a-f0-9]{{32}}"
        for runner in runners:
            name = runner["name"]
            if not re.fullmatch(pattern, name) or name in live_names:
                continue
            if not runner["busy"] and name not in running:
                self._github.retire(runner["id"])

        # Fixed names enforce the cap even after a timeout or overlapping revision.
        needed = max(0, demand - available)
        started = 0
        for slot in slots:
            if started >= needed:
                break
            if slot in occupied:
                continue
            name = f"{slot}-{uuid.uuid4().hex}"
            jit = self._github.register(name)
            self._compute.create(slot, name, jit["encoded_jit_config"])
            started += 1
        return {"queued": demand, "occupied": len(occupied), "started": started}
