"""Full Pi -> RPC -> fake companion -> recorder -> private importer acceptance."""

import asyncio
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from vla_platform.collection_api import ROOT, CollectionInput, CollectionRuns
from vla_platform.datasets.library import DatasetLibrary


class PiCollectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pi-collection-")
        self.root = Path(self.temp.name)
        self.app = SimpleNamespace(
            state=SimpleNamespace(robot_lab=object(), dataset_library=DatasetLibrary(self.root))
        )
        self.service = CollectionRuns(self.app, self.root)

    async def asyncTearDown(self):
        await self.service.close()
        self.temp.cleanup()

    @unittest.skipUnless(
        (ROOT / "workers/pi_collection/node_modules/@earendil-works/pi-agent-core").is_dir(),
        "Full Pi lane requires Node 24 and npm ci in workers/pi_collection",
    )
    async def test_complete_pi_episode_and_reload_export(self):
        run = await self.service.start("test-project", CollectionInput())
        await asyncio.wait_for(self.service.tasks[run["id"]], 60)
        saved = self.service.get(run["id"])
        self.assertEqual(saved["status"], "saved", saved)
        self.assertGreater(saved["frames"], 20)
        self.assertIsNotNone(saved["dataset"], saved)
        trace = [
            json.loads(x)
            for x in (self.service.directory(run["id"]) / "trace.jsonl").read_text().splitlines()
        ]
        commands = [e for e in trace if e["type"] == "command_result"]
        self.assertEqual(len(commands), 2, trace)
        self.assertTrue(all(e["receipt"]["measured_completion_verified"] for e in commands))
        reloaded = CollectionRuns(self.app, self.root)
        self.assertEqual(reloaded.get(run["id"]), saved)
        archive = self.app.state.dataset_library.export(saved["dataset"]["id"])
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            self.assertTrue(any(n.endswith("evidence/frames.jsonl") for n in names), names)
            self.assertTrue(any(n.startswith("raw/") for n in names), names)
            info = json.loads(bundle.read(next(n for n in names if n.endswith("meta/info.json"))))
            self.assertIn("observation.images.wrist", info["features"])
            self.assertIn("observation.images.side", info["features"])
            frames = [
                json.loads(x)
                for x in bundle.read(
                    next(n for n in names if n.endswith("evidence/frames.jsonl"))
                ).splitlines()
            ]
            control = [
                json.loads(x)
                for x in bundle.read(
                    next(n for n in names if n.endswith("evidence/controller.jsonl"))
                ).splitlines()
            ]
            sample_ids = {x["sample_id"] for x in control}
            self.assertTrue(all(f["controller_sample_id"] in sample_ids for f in frames))
            self.assertTrue(any(f["command_id"] == commands[0]["command_id"] for f in frames))
        archive.unlink()

    @unittest.skipUnless(
        (ROOT / "workers/pi_collection/node_modules/@earendil-works/pi-agent-core").is_dir(),
        "Full Pi lane requires Node 24 and npm ci in workers/pi_collection",
    )
    async def test_stop_preserves_partial(self):
        run = await self.service.start("test-project", CollectionInput())
        for _ in range(100):
            if self.service.get(run["id"]).get("frames", 0) >= 3:
                break
            await asyncio.sleep(0.1)
        await self.service.stop(run["id"])
        await asyncio.wait_for(self.service.tasks[run["id"]], 30)
        value = self.service.get(run["id"])
        self.assertIn(value["status"], {"stopped", "failed"}, value)
        self.assertGreaterEqual(value["frames"], 2, value)
        self.assertIsNotNone(value["dataset"], value)

    async def test_missing_camera_fails_without_dataset(self):
        run = await self.service.start("test-project", CollectionInput(camera_indices=[0, 2]))
        await asyncio.wait_for(self.service.tasks[run["id"]], 30)
        value = self.service.get(run["id"])
        self.assertEqual(value["status"], "failed", value)
        self.assertIsNone(value["dataset"])

    async def test_immediate_cancel_never_starts_a_companion(self):
        run = await self.service.start("test-project", CollectionInput())
        await self.service.stop(run["id"])
        await asyncio.gather(self.service.tasks[run["id"]], return_exceptions=True)
        self.assertEqual(self.service.get(run["id"])["status"], "stopped")
        self.assertFalse((self.service.directory(run["id"]) / "mock-companion").exists())

    async def test_restart_marks_interrupted_without_resuming(self):
        run = await self.service.start("test-project", CollectionInput())
        await self.service.stop(run["id"])
        self.service.update(run["id"], status="running")
        reloaded = CollectionRuns(self.app, self.root)
        self.assertEqual(reloaded.get(run["id"])["status"], "interrupted")
        self.assertEqual(reloaded.live, {})

    async def test_physical_requires_confirmation(self):
        with self.assertRaisesRegex(ValueError, "Confirm"):
            await self.service.start("test-project", CollectionInput(mode="physical"))


if __name__ == "__main__":
    unittest.main()
