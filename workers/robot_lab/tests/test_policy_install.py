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
from test_smolvla_contract import SmolVLAContractTests


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

    def with_files(self, extra, artifact=None):
        """The same artifact with more checkpoint files, listed in its manifest."""
        descriptor, entries = artifact or self.artifact()
        descriptor, entries = dict(descriptor), list(entries)
        manifest = json.loads(entries[0][1])
        for name, data in extra.items():
            path = "checkpoint/pretrained_model/" + name
            manifest["files"][path] = hashlib.sha256(data).hexdigest()
            entries.append(("policy/" + path, data, tarfile.REGTYPE))
        document = json.dumps(manifest).encode()
        entries[0] = ("policy/manifest.json", document, tarfile.REGTYPE)
        descriptor["source_manifest_sha256"] = hashlib.sha256(document).hexdigest()
        return descriptor, entries

    def smolvla_artifact(self):
        """A SmolVLA checkpoint as current LeRobot saves it: its tokenizer in a folder,
        named as an artifact of the tokenizer step, beside the backbone's data files."""
        source = SmolVLAContractTests("test_closed_dispatch_and_exact_native_processors")
        source.setUp()
        self.addCleanup(source.tearDown)
        pipeline = json.loads((source.checkpoint / "policy_preprocessor.json").read_text())
        pipeline["steps"][3]["config"]["tokenizer_name"] = "tokenizer"
        pipeline["steps"][3]["artifacts"] = {"tokenizer_name": "tokenizer"}
        (source.checkpoint / "policy_preprocessor.json").write_text(json.dumps(pipeline))
        folder = source.checkpoint / "tokenizer"
        folder.mkdir()
        (folder / "tokenizer.json").write_text("{}")
        (folder / "tokenizer_config.json").write_text("{}")
        (folder / "chat_template.jinja").write_text("template")
        files = {
            "checkpoint/pretrained_model/" + path.relative_to(source.checkpoint).as_posix(): (
                path.read_bytes()
            )
            for path in sorted(source.checkpoint.rglob("*"))
            if path.is_file()
        }
        manifest = json.dumps(
            {
                "schema_version": 1,
                "metadata": {},
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
            }
        ).encode()
        descriptor = {
            "schema_version": 1,
            "id": "installed-duck",
            "name": "Duck",
            "policy_type": "smolvla",
            "fps": 23,
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

    def test_a_smolvla_checkpoint_installs_with_the_tokenizer_folder_it_was_saved_with(self):
        """LeRobot saves ``tokenizer/`` beside the weights; that used to refuse the install."""
        descriptor, entries = self.smolvla_artifact()
        archive = self.tar(entries)
        result = install_archive(self.lab, io.BytesIO(archive), len(archive), descriptor)
        self.assertEqual(result["installed_policy_id"], "installed-duck")
        installed = self.lab.policy.root / "installed-duck/pretrained_model"
        self.assertEqual((installed / "tokenizer/chat_template.jinja").read_text(), "template")
        self.assertTrue((installed / "assets/smolvlm/tokenizer.json").is_file())
        listed = self.lab.policy.catalog()["policies"]
        self.assertTrue(
            next(item for item in listed if item["id"] == "installed-duck")["compatible"]
        )
        self.assertIsNone(self.lab.policy.runtime)
        # Its bytes are covered by the manifest: a changed copy refuses the install.
        changed = [
            (name, data + b" " if name.endswith("tokenizer/tokenizer.json") else data, kind)
            for name, data, kind in entries
        ]
        archive = self.tar(changed)
        with self.assertRaises(ValueError):
            install_archive(
                self.lab, io.BytesIO(archive), len(archive), descriptor | {"id": "second"}
            )
        self.assertFalse((self.lab.policy.root / "second").exists())

    def test_a_tokenizer_folder_is_not_accepted_for_a_policy_that_has_no_tokenizer(self):
        descriptor, entries = self.with_files({"tokenizer/tokenizer.json": b"{}"})
        archive = self.tar(entries)
        with self.assertRaisesRegex(ValueError, "Only native JSON"):
            install_archive(self.lab, io.BytesIO(archive), len(archive), descriptor)
        self.assertFalse((self.lab.policy.root / "installed-pen").exists())

    def test_other_nested_checkpoint_files_are_still_refused(self):
        artifact = self.artifact()
        for name in (
            "extra/file.json",
            "tokenizer/deeper/file.json",
            "tokenizer/code.py",
            "assets/other/config.json",
        ):
            descriptor, entries = self.with_files({name: b"{}"}, artifact)
            archive = self.tar(entries)
            with self.assertRaisesRegex(ValueError, "unsupported nested asset"):
                install_archive(self.lab, io.BytesIO(archive), len(archive), descriptor)
            self.assertFalse((self.lab.policy.root / "installed-pen").exists())

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
