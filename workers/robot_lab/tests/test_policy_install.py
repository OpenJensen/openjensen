"""Artifact TAR integrity and atomic publication without any motor or model inference."""

import hashlib
import io
import json
import tarfile
import threading
import time
import unittest

import test_policy as fixtures
from policy_install import install_archive


class InstallTests(unittest.TestCase):
    setUp = fixtures.PolicyTests.setUp
    tearDown = fixtures.PolicyTests.tearDown
    install = fixtures.PolicyTests.install

    def artifact(self):
        runner, _, checkpoint = self.install()
        files = {
            "checkpoint/pretrained_model/" + path.name: path.read_bytes()
            for path in checkpoint.iterdir()
        }
        files["checkpoint/recipe.json"] = b'{"fps":13}'
        manifest = json.dumps(
            {
                "schema_version": 1,
                "metadata": {},
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
            }
        ).encode()
        descriptor = dict(runner.catalog()["policies"][0])
        descriptor = {
            "schema_version": 1,
            "id": "installed-pen",
            "name": "Pen",
            "policy_type": "act",
            "fps": 13,
            "camera_key": None,
            "joint_names": list(fixtures.JOINTS),
            "state_units": fixtures.UNITS,
            "action_units": fixtures.UNITS,
            "checkpoint": "pretrained_model",
            "source_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        }
        entries = [("policy/manifest.json", manifest, tarfile.REGTYPE)]
        entries.extend(("policy/" + name, data, tarfile.REGTYPE) for name, data in files.items())
        return descriptor, entries

    def tar(self, entries):
        result = io.BytesIO()
        with tarfile.open(fileobj=result, mode="w") as archive:
            for name, data, kind in entries:
                member = tarfile.TarInfo(name)
                member.size, member.type = len(data), kind
                if kind == tarfile.SYMTYPE:
                    member.linkname = "/tmp/outside"
                    member.size = 0
                archive.addfile(member, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
        return result.getvalue()

    def test_verified_native_archive_installs_atomically_and_does_not_load_or_arm(self):
        descriptor, entries = self.artifact()
        archive = self.tar(entries)
        result = install_archive(self.lab, io.BytesIO(archive), len(archive), descriptor)
        self.assertEqual(result["installed_policy_id"], "installed-pen")
        self.assertEqual(self.lab.mode, "idle")
        self.assertEqual(self.buses, [])
        self.assertIsNone(self.lab.policy.runtime)
        self.assertTrue(
            (self.lab.policy.root / "installed-pen/pretrained_model/model.safetensors").is_file()
        )
        self.assertTrue((self.lab.policy.root / "pen").is_dir())
        self.assertFalse(list(self.lab.data_dir.glob(".policy-install-*")))

    def test_paths_links_duplicate_hash_and_manifest_damage_never_publish(self):
        descriptor, entries = self.artifact()
        damaged = list(entries)
        name, data, kind = damaged[-1]
        damaged[-1] = (name, data + b"changed", kind)
        for candidate in (
            entries + [("policy/../../outside", b"x", tarfile.REGTYPE)],
            entries + [("policy/link", b"", tarfile.SYMTYPE)],
            entries + [entries[0]],
            damaged,
        ):
            archive = self.tar(candidate)
            with self.assertRaises(ValueError):
                install_archive(self.lab, io.BytesIO(archive), len(archive), descriptor)
            self.assertFalse((self.lab.policy.root / "installed-pen").exists())
        archive = self.tar(entries)
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            install_archive(
                self.lab,
                io.BytesIO(archive),
                len(archive),
                descriptor | {"source_manifest_sha256": "0" * 64},
            )
        self.assertTrue((self.lab.policy.root / "pen").is_dir())
        self.assertFalse(list(self.lab.data_dir.glob(".policy-install-*")))

    def test_stop_releases_blocked_upload_and_new_install_fences_old_completion(self):
        descriptor, entries = self.artifact()
        archive = self.tar(entries)
        entered, release = threading.Event(), threading.Event()
        errors = []

        class Blocked(io.BytesIO):
            def read(self, size=-1):
                entered.set()
                release.wait(timeout=3)
                return super().read(size)

        def old_install():
            try:
                install_archive(self.lab, Blocked(archive), len(archive), descriptor)
            except ValueError as error:
                errors.append(str(error))

        thread = threading.Thread(target=old_install)
        thread.start()
        self.assertTrue(entered.wait(timeout=1))
        before = time.monotonic()
        self.lab.stop()
        self.assertLess(time.monotonic() - before, 0.5)
        result = install_archive(self.lab, io.BytesIO(archive), len(archive), descriptor)
        self.assertEqual(result["installed_policy_id"], "installed-pen")
        release.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(any("cancelled" in error for error in errors))
        self.assertTrue((self.lab.policy.root / "installed-pen").is_dir())
        self.assertFalse(list(self.lab.data_dir.glob(".policy-install-*")))

    def test_stop_cancels_an_in_progress_install_before_publication(self):
        descriptor, entries = self.artifact()
        archive = self.tar(entries)
        lab = self.lab

        class Interrupted(io.BytesIO):
            def read(self, size=-1):
                lab.stop()
                return super().read(size)

        with self.assertRaisesRegex(ValueError, "cancelled"):
            install_archive(self.lab, Interrupted(archive), len(archive), descriptor)
        self.assertFalse((self.lab.policy.root / "installed-pen").exists())
        self.assertEqual(self.lab.mode, "idle")
        self.assertNotEqual(self.lab.policy.phase, "installing")


if __name__ == "__main__":
    unittest.main()
