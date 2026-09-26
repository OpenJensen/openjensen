"""Resolve scene files after a job bundle moves from the client to a worker."""

from pathlib import Path
import shutil
import tempfile
import unittest

import yaml

from sim_worker.adapters.manifest import load


_DEMO = Path(__file__).resolve().parents[1] / "demo.yaml"
_SUFFIXES = (".usd", ".usda", ".usdc")


class ScenePathTests(unittest.TestCase):
    def _manifest(self, path, scene):
        data = yaml.safe_load(_DEMO.read_text())
        data["scene"]["uri"] = scene
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data))
        return path

    def test_relative_scene_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = root / "client" / "job bundle"
            scene = client / "assets" / "my scene.usda"
            scene.parent.mkdir(parents=True)
            scene.write_text("#usda 1.0\n")
            manifest = self._manifest(client / "job.yaml", "assets/my scene.usda")
            self.assertEqual(load(manifest).scene, str(scene.resolve()))

            # Copy the bundle as SkyPilot does; resolve against its new location.
            worker = root / "worker" / "job bundle"
            shutil.copytree(client, worker)
            resolved = load(worker / "job.yaml").scene
            self.assertEqual(resolved, str((worker / "assets/my scene.usda").resolve()))
            self.assertEqual(Path(resolved).read_bytes(), scene.read_bytes())

    def test_relative_usd_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in _SUFFIXES:
                with self.subTest(suffix=suffix):
                    scene = root / f"scene{suffix}"
                    scene.write_bytes(b"USD fixture")
                    manifest = self._manifest(root / "job.yaml", scene.name)
                    self.assertEqual(load(manifest).scene, str(scene.resolve()))

    def test_missing_relative_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory) / "job.yaml", "missing.usda")
            with self.assertRaisesRegex(ValueError, "Scene file not found"):
                load(manifest)

    def test_scene_directory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scene.usda").mkdir()
            manifest = self._manifest(root / "job.yaml", "scene.usda")
            with self.assertRaisesRegex(ValueError, "Scene file not found"):
                load(manifest)

    def test_scene_urls_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.yaml"
            for scheme in ("https", "gs", "omniverse", "file"):
                with self.subTest(scheme=scheme):
                    manifest = self._manifest(path, f"{scheme}://host/scene.usda")
                    with self.assertRaisesRegex(ValueError, "USD file path"):
                        load(manifest)

    def test_absolute_path_kept(self):
        # Absolute paths may only exist in the target container, not on the client.
        scene = "/inputs/custom/scene.usda"
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory) / "job.yaml", scene)
            self.assertEqual(load(manifest).scene, scene)


if __name__ == "__main__":
    unittest.main()
