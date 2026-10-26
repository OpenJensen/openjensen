"""Authority races during acquisition, cancellation, shutdown, and restart."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from test_agent_simulation_service import FakeTeaching
from vla_platform.agent_control.contracts import PlanRequest, SessionCreate
from vla_platform.agent_control.service import AgentError
from vla_platform.api import create_app
from vla_platform.settings import Settings


class BlockedTeaching(FakeTeaching):
    def __init__(self, phase):
        super().__init__()
        self.epoch = 7
        self.phase = phase
        self.entered = asyncio.Event()
        self.proceed = asyncio.Event()

    async def blocked(self):
        self.entered.set()
        try:
            await self.proceed.wait()
        except asyncio.CancelledError:
            # A dispatched network call can return after cancellation was requested.
            await self.proceed.wait()

    async def relay(self, project, job, path):
        if self.phase == "state":
            await self.blocked()
        return await super().relay(project, job, path)

    async def agent_relay(self, project, job, path, payload=None):
        result = await super().agent_relay(project, job, path, payload)
        if payload is not None and path == "/agent/" + self.phase:
            await self.blocked()
        return result


@pytest.fixture
def service(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        project = client.post("/api/v1/projects", json={"name": "session races"}).json()["id"]
        yield client, client.app.state.agent_control, project


async def eventually(check):
    async with asyncio.timeout(3):
        while not await check():
            await asyncio.sleep(0.005)


def teaching_request():
    return SessionCreate(
        task="shoulder_pan +0.03 rad",
        target_kind="teaching",
        teaching_job_id="job_fixture",
        clock_policy="stepped",
    )


@pytest.mark.parametrize("phase", ["state", "acquire", "observe"])
def test_stop_during_acquisition_cannot_publish_ready_or_reacquire(service, phase):
    client, agent, project = service

    async def scenario():
        target = BlockedTeaching(phase)
        agent.execution.teaching = target
        creation = asyncio.create_task(agent.create(project, teaching_request()))
        stopping = None
        try:
            await asyncio.wait_for(target.entered.wait(), 3)
            sessions = await agent.list(project)
            assert len(sessions) == 1 and sessions[0].status == "observing"
            identity = sessions[0].id
            revoked_epoch = sessions[0].authority_epoch + 1
            assert agent.pending
            stopping = asyncio.create_task(agent.interrupt(identity, "stop"))

            async def revoked():
                current = await agent.get(identity)
                return current.authority_epoch == revoked_epoch and current.status in {
                    "stopped",
                    "interrupted",
                }

            await eventually(revoked)
            target.proceed.set()
            outcomes = await asyncio.wait_for(
                asyncio.gather(creation, stopping, return_exceptions=True), 3
            )
            if not isinstance(outcomes[0], (AgentError, asyncio.CancelledError)):
                assert outcomes[0].status in {"stopped", "interrupted"}
            current, private = await agent.load(identity)
            assert current.status in {"stopped", "interrupted"}
            assert current.authority_epoch == revoked_epoch and current.observation is None
            assert current.pending_proposal is None and target.executions == []
            assert target.owner is None
            acquisitions = [p for operation, p in target.commands if operation == "acquire"]
            if phase == "state":
                assert acquisitions == []
                assert target.release_count == 0
            else:
                assert len(acquisitions) == 1 and target.release_count == 1
                releases = [p for operation, p in target.commands if operation == "release"]
                assert releases[-1]["authority_epoch"] == 8
                assert private.get("released") is True
            assert not any(event.kind == "observed" for event in await agent.events(identity, 0))
        finally:
            target.proceed.set()
            for task in (creation, stopping):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*(t for t in (creation, stopping) if t), return_exceptions=True)

    client.portal.call(scenario)


def test_cancelled_create_cleans_up_acquired_owner_with_actual_epoch(service):
    client, agent, project = service

    async def scenario():
        target = BlockedTeaching("observe")
        agent.execution.teaching = target
        creation = asyncio.create_task(agent.create(project, teaching_request()))
        try:
            await asyncio.wait_for(target.entered.wait(), 3)
            identity = (await agent.list(project))[0].id
            assert target.owner == identity and target.epoch == 8
            creation.cancel()
            await asyncio.sleep(0.01)
            target.proceed.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(creation, 3)

            async def drained():
                return target.owner is None and all(task.done() for task in agent.tasks.values())

            await eventually(drained)
            current, private = await agent.load(identity)
            assert current.status in {"stopped", "interrupted", "error"}
            assert current.observation is None and current.pending_proposal is None
            assert private.get("released") is True and target.release_count == 1
            releases = [p for operation, p in target.commands if operation == "release"]
            assert releases[-1]["authority_epoch"] == 8
            assert target.executions == []
        finally:
            target.proceed.set()
            if not creation.done():
                creation.cancel()
            await asyncio.gather(creation, return_exceptions=True)

    client.portal.call(scenario)


def test_shutdown_drains_tasks_even_when_session_is_already_terminal(service):
    client, agent, project = service

    async def scenario():
        current = await agent.create(project, SessionCreate(task="inspect"))
        current, private = await agent.load(current.id)
        current.status = "stopped"
        await agent.save(current, private, "test_terminal")
        started, drained = asyncio.Event(), asyncio.Event()

        async def lingering():
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0.01)
                drained.set()

        task = asyncio.create_task(lingering())
        agent.tasks[current.id] = task
        await started.wait()
        try:
            await asyncio.wait_for(agent.close(), 3)
            assert drained.is_set() and task.done()
            assert not agent.pending
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    client.portal.call(scenario)


def test_restart_preserves_original_started_execution_identity(service):
    client, agent, project = service

    async def scenario():
        current = await agent.create(project, SessionCreate(task="shoulder_pan +0.03 rad"))
        await agent.propose(current.id, PlanRequest(request_id="before_restart"))
        await agent.tasks[current.id]
        current, private = await agent.load(current.id)
        assert current.status == "waiting_for_acceptance"
        proposal_id, epoch = current.pending_proposal.id, current.authority_epoch
        current.status = "executing"
        current.pending_proposal.status = "started"
        private["execution_id"] = "original_execution_identity"
        await agent.save(current, private, "execution_started")
        await agent.reconcile()
        recovered = await agent.get(current.id)
        assert recovered.status == "interrupted" and recovered.authority_epoch == epoch + 1
        receipt = recovered.receipts[-1]
        assert receipt.id == "original_execution_identity" and receipt.status == "unknown"
        assert receipt.proposal_id == proposal_id and receipt.applied_targets == {}
        await agent.reconcile()
        assert len((await agent.get(current.id)).receipts) == 1
        with pytest.raises(AgentError):
            await agent.propose(current.id, PlanRequest(request_id="after_restart"))
        assert (await agent.get(current.id)).calls_used == 1

    client.portal.call(scenario)


def test_result_after_swallowed_deadline_cancellation_cannot_create_proposal(service):
    client, agent, project = service

    async def scenario():
        timeouts = []

        async def late(profile, payload):
            timeouts.append(payload["timeout_seconds"])
            async with asyncio.timeout(payload["timeout_seconds"]):
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    pass
            return {
                "status": "proposed",
                "model": "deterministic-mock-v1",
                "arguments": {"joint": "shoulder_pan", "delta_rad": 0.03},
                "usage": {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
                "continuation": None,
            }

        agent.planner = late
        current = await agent.create(
            project, SessionCreate(task="shoulder_pan +0.03 rad", max_seconds=1)
        )
        await agent.propose(current.id, PlanRequest(request_id="late_budget"))
        await asyncio.wait_for(agent.tasks[current.id], 3)
        rejected = await agent.get(current.id)
        assert rejected.status == "error" and rejected.pending_proposal is None
        assert (
            rejected.receipts == [] and rejected.input_tokens == 3 and rejected.output_tokens == 5
        )
        assert len(timeouts) == 1 and 0 < timeouts[0] < 1
        assert any(
            event.kind == "budget_result_rejected" for event in await agent.events(current.id, 0)
        )
        with pytest.raises(AgentError):
            await agent.propose(current.id, PlanRequest(request_id="no_retry"))

    client.portal.call(scenario)
