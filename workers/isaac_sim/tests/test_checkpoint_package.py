"""Real safetensors containers and adversarial package inputs; no ML imports."""

import json
import struct
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_worker.rollout.checkpoint_package import resolve_checkpoint


def tensor_file(path, tensors=None):
    tensors = tensors or {"weight": ([1], [1.0])}
    header, data = {}, b""
    for name, (shape, values) in tensors.items():
        raw = struct.pack("<" + "f" * len(values), *values)
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def export(root, family="act"):
    root.mkdir(parents=True, exist_ok=True)
    features = {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.front": {"type": "VISUAL", "shape": [3, 32, 32]},
    }
    outputs = {"action": {"type": "ACTION", "shape": [6]}}
    config = {
        "type": family,
        "n_obs_steps": 1,
        "chunk_size": 100 if family == "act" else 50,
        "input_features": features,
        "output_features": outputs,
    }
    (root / "config.json").write_text(json.dumps(config))
    tensor_file(root / "model.safetensors")
    for name, registry, required in [
        ("preprocessor", "normalizer_processor", features),
        ("postprocessor", "unnormalizer_processor", outputs),
    ]:
        stats = {}
        for key, feature in required.items():
            shape = [3, 1, 1] if feature["type"] == "VISUAL" else feature["shape"]
            count = 3 if feature["type"] == "VISUAL" else 6
            stats[key + ".mean"] = (shape, [0.0] * count)
            stats[key + ".std"] = (shape, [1.0] * count)
        tensor_file(root / (name + ".safetensors"), stats)
        steps = [
            {
                "registry_name": registry,
                "config": {
                    "features": required,
                    "norm_map": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD", "VISUAL": "MEAN_STD"},
                },
                "state_file": name + ".safetensors",
            }
        ]
        if family == "smolvla" and name == "preprocessor":
            steps.insert(0, {"registry_name": "smolvla_new_line_processor", "config": {}})
            steps.insert(
                0,
                {
                    "registry_name": "tokenizer_processor",
                    "config": {"tokenizer_name": "approved/example"},
                },
            )
        (root / ("policy_" + name + ".json")).write_text(
            json.dumps({"name": "policy_" + name, "steps": steps})
        )
    return root


def manifest(root, detailed=False):
    import hashlib

    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != root / "manifest.json":
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            files[path.relative_to(root).as_posix()] = (
                {"sha256": sha, "bytes": len(raw)} if detailed else sha
            )
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}))


def archive(source, dest):
    with tarfile.open(dest, "w") as tar:
        tar.add(source, arcname="policy")


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_both_flat_exports_are_copied_and_cleaned(self):
        for family in ("act", "smolvla"):
            source = export(self.root / family, family)
            (source / "README.md").write_text("unknown harmless documentation")
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            with resolve_checkpoint(source) as result:
                selected = result.directory
                self.assertEqual(result.checkpoint.policy_type, family)
                self.assertNotEqual(selected, source)
                self.assertIsNone(result.source_sha256)
                self.assertEqual(
                    (selected / "README.md").read_text(), "unknown harmless documentation"
                )
                self.assertFalse(result.metadata()["runtime_verified"])
            self.assertFalse(selected.exists())
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_firebird_nested_archive_both_families(self):
        for family in ("act", "smolvla"):
            wrapper = self.root / family
            export(wrapper / "policy", family)
            manifest(wrapper / "policy", True)
            (wrapper / "lineage.json").write_text("{}")
            manifest(wrapper)
            path = self.root / (family + ".tar")
            archive(wrapper, path)
            with resolve_checkpoint(path, archive=True) as result:
                self.assertEqual(result.checkpoint.policy_type, family)
                self.assertEqual(len(result.manifests), 2)
                self.assertEqual(len(result.source_sha256), 64)

    def test_exception_cleans_private_copy(self):
        with self.assertRaisesRegex(RuntimeError, "cancel"):
            with resolve_checkpoint(export(self.root / "model")) as result:
                selected = result.directory
                raise RuntimeError("cancel")
        self.assertFalse(selected.exists())

    def test_missing_stats_and_bare_weights(self):
        source = export(self.root / "model")
        (source / "preprocessor.safetensors").unlink()
        with self.assertRaises(ValueError):
            with resolve_checkpoint(source):
                self.fail("accepted missing statistics")
        with self.assertRaises(ValueError):
            with resolve_checkpoint(source / "model.safetensors"):
                self.fail("accepted bare weights")

    def test_unknown_processor_rejected(self):
        source = export(self.root / "model")
        path = source / "policy_preprocessor.json"
        data = json.loads(path.read_text())
        data["steps"][0]["registry_name"] = "custom_executable"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "processor"):
            with resolve_checkpoint(source):
                self.fail("accepted custom processor")

    def test_manifest_tampering_and_omissions(self):
        source = export(self.root / "model")
        manifest(source)
        (source / "README.md").write_text("unlisted")
        with self.assertRaisesRegex(ValueError, "manifest"):
            with resolve_checkpoint(source):
                self.fail("accepted unlisted file")
        (source / "README.md").unlink()
        (source / "config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest"):
            with resolve_checkpoint(source):
                self.fail("accepted tampering")

    def test_rejects_multiple_policies(self):
        export(self.root / "wrapper" / "policy")
        export(self.root / "wrapper" / "pretrained_model", "smolvla")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            with resolve_checkpoint(self.root / "wrapper"):
                self.fail("ambiguous package")

    def test_symlinks_and_special_files(self):
        source = export(self.root / "model")
        (source / "linked").symlink_to(source / "config.json")
        with self.assertRaisesRegex(ValueError, "regular|symlink"):
            with resolve_checkpoint(source):
                self.fail("accepted symlink")

    def test_archive_traversal_links_duplicate_and_special(self):
        cases = [
            ("../escape", tarfile.REGTYPE),
            ("/absolute", tarfile.REGTYPE),
            ("C:/drive", tarfile.REGTYPE),
            ("a\\b", tarfile.REGTYPE),
            ("policy/link", tarfile.SYMTYPE),
            ("policy/hard", tarfile.LNKTYPE),
            ("policy/fifo", tarfile.FIFOTYPE),
        ]
        for index, (name, kind) in enumerate(cases):
            path = self.root / f"bad{index}.tar"
            with tarfile.open(path, "w") as tar:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.linkname = "target"
                tar.addfile(info)
            with self.subTest(name=name), self.assertRaises(ValueError):
                with resolve_checkpoint(path, archive=True):
                    self.fail("unsafe archive")
        path = self.root / "duplicate.tar"
        with tarfile.open(path, "w") as tar:
            for _ in range(2):
                tar.addfile(tarfile.TarInfo("same"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            with resolve_checkpoint(path, archive=True):
                self.fail("duplicate")

    def test_size_and_member_limits(self):
        source = export(self.root / "model")
        with patch("sim_worker.rollout.checkpoint_package.MAX_FILES", 2):
            with self.assertRaisesRegex(ValueError, "limit"):
                with resolve_checkpoint(source):
                    self.fail("file limit")
        with patch("sim_worker.rollout.checkpoint_package.MAX_EXPANDED", 100):
            with self.assertRaisesRegex(ValueError, "limit"):
                with resolve_checkpoint(source):
                    self.fail("size limit")

    def test_fake_safetensors_or_missing_normalization_values(self):
        source = export(self.root / "model")
        (source / "model.safetensors").write_bytes(b"not a model")
        with self.assertRaisesRegex(ValueError, "safetensors"):
            with resolve_checkpoint(source):
                self.fail("invalid weights")
        tensor_file(source / "model.safetensors")
        tensor_file(source / "preprocessor.safetensors")
        with self.assertRaisesRegex(ValueError, "statistics"):
            with resolve_checkpoint(source):
                self.fail("missing feature statistics")

    def test_json_duplicate_nonfinite_and_non_utf8(self):
        source = export(self.root / "model")
        for value in (
            b'{"type":"act","type":"smolvla"}',
            b'{"unused":1e309}',
            "{}".encode("utf-16"),
        ):
            (source / "config.json").write_bytes(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                with resolve_checkpoint(source):
                    self.fail("invalid JSON")

    def test_gzip_and_persistent_import(self):
        from sim_worker.rollout.checkpoint_package import import_checkpoint

        source = export(self.root / "smol", "smolvla")
        archive_path = self.root / "model.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(source, arcname="policy")
        output, receipt = self.root / "imported", self.root / "receipt.json"
        result = import_checkpoint(archive_path, output, receipt, archive=True)
        self.assertEqual(result["checkpoint"]["policy_type"], "smolvla")
        self.assertEqual(json.loads(receipt.read_text()), result)
        self.assertTrue((output / result["directory"] / "model.safetensors").is_file())
        self.assertNotIn("/private/", receipt.read_text())
        with self.assertRaisesRegex(ValueError, "already exist"):
            import_checkpoint(archive_path, output, receipt, archive=True)

    def test_import_failure_preserves_existing_receipt_and_source(self):
        from sim_worker.rollout.checkpoint_package import import_checkpoint

        source = export(self.root / "source")
        output, receipt = self.root / "imported", self.root / "receipt.json"
        receipt.write_text("existing")
        with self.assertRaises(ValueError):
            import_checkpoint(source, output, receipt)
        self.assertEqual(receipt.read_text(), "existing")
        self.assertFalse(output.exists())
        receipt.unlink()
        with patch(
            "sim_worker.rollout.checkpoint_package.shutil.copytree",
            side_effect=OSError("copy failed"),
        ):
            with self.assertRaises(OSError):
                import_checkpoint(source, output, receipt)
        self.assertFalse(output.exists())
        self.assertFalse(receipt.exists())
        self.assertTrue((source / "model.safetensors").is_file())

    def test_oversized_pax_metadata_fails_before_allocation(self):
        path = self.root / "pax.tar"
        info = tarfile.TarInfo("pax")
        info.type = tarfile.XHDTYPE
        info.size = 10**9
        path.write_bytes(info.tobuf() + b"\0" * 1024)
        with self.assertRaisesRegex(ValueError, "metadata size limit"):
            with resolve_checkpoint(path, archive=True):
                self.fail("huge PAX header")

    def test_oversized_processor_json_bounded_before_inspector(self):
        source = export(self.root / "source")
        (source / "policy_preprocessor.json").write_bytes(b" " * 10000)
        with patch("sim_worker.rollout.checkpoint_package.MAX_JSON", 4096):
            with self.assertRaisesRegex(ValueError, "JSON size limit"):
                with resolve_checkpoint(source):
                    self.fail("oversized JSON")

    def test_stats_nonfinite_and_negative_std_rejected(self):
        source = export(self.root / "source")
        for value in (float("nan"), -1.0):
            stats = {"action.mean": ([6], [0.0] * 6), "action.std": ([6], [value] * 6)}
            tensor_file(source / "postprocessor.safetensors", stats)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "statistics values"):
                with resolve_checkpoint(source):
                    self.fail("invalid stats")

    def test_directory_file_growth_is_bounded_and_rejected(self):
        source = export(self.root / "source")
        from sim_worker.rollout import checkpoint_package as package

        real_copy = package._copy

        def grow(stream, target, size):
            result = real_copy(stream, target, size)
            if target.name == "model.safetensors":
                with (source / "model.safetensors").open("ab") as append:
                    append.write(b"changed")
            return result

        with patch.object(package, "_copy", side_effect=grow):
            with self.assertRaisesRegex(ValueError, "changed"):
                with resolve_checkpoint(source):
                    self.fail("concurrent mutation")

    def test_directory_symlink_root_rejected(self):
        source = export(self.root / "source")
        link = self.root / "link"
        link.symlink_to(source, target_is_directory=True)
        with self.assertRaises(ValueError):
            with resolve_checkpoint(link):
                self.fail("symlink root")

    def test_safetensors_overlap_and_truncation_rejected(self):
        source = export(self.root / "source")
        tensor_file(source / "model.safetensors", {"a": ([1], [1.0]), "b": ([1], [2.0])})
        raw = (source / "model.safetensors").read_bytes()
        (source / "model.safetensors").write_bytes(raw[:-1])
        with self.assertRaisesRegex(ValueError, "safetensors"):
            with resolve_checkpoint(source):
                self.fail("truncated tensor")

    def test_cli_import_and_inspection_without_ml_imports(self):
        import subprocess
        import sys

        source = export(self.root / "source", "smolvla")
        output, receipt = self.root / "output", self.root / "receipt.json"
        command = [
            sys.executable,
            "-m",
            "sim_worker.rollout.checkpoint_package",
            "--source",
            str(source),
            "--output-dir",
            str(output),
            "--json-output",
            str(receipt),
        ]
        done = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout), json.loads(receipt.read_text()))
        inspect = subprocess.run(
            [
                sys.executable,
                "-m",
                "sim_worker.rollout.checkpoint_package",
                "--checkpoint",
                str(source),
                "--inspect-only",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(inspect.returncode, 0, inspect.stderr)
        self.assertEqual(json.loads(inspect.stdout)["checkpoint"]["policy_type"], "smolvla")

    def test_receipt_failure_removes_owned_partial_import(self):
        from sim_worker.rollout.checkpoint_package import import_checkpoint

        source = export(self.root / "source")
        output, receipt = self.root / "output", self.root / "receipt.json"
        with patch(
            "sim_worker.rollout.checkpoint_package.os.fsync", side_effect=OSError("disk error")
        ):
            with self.assertRaises(OSError):
                import_checkpoint(source, output, receipt)
        self.assertFalse(output.exists())
        self.assertFalse(receipt.exists())

    def test_archive_declared_member_limit_without_payload(self):
        path = self.root / "large.tar"
        info = tarfile.TarInfo("model.safetensors")
        info.size = 3 * 1024**3
        path.write_bytes(info.tobuf() + b"\0" * 1024)
        with self.assertRaisesRegex(ValueError, "size limit"):
            with resolve_checkpoint(path, archive=True):
                self.fail("huge file")

    def test_launch_archive_keeps_copy_until_submission_returns(self):
        import importlib.util
        import os
        import sys
        from contextlib import contextmanager

        launch_root = Path(__file__).resolve().parents[2] / "skypilot"
        sys.path.insert(0, str(launch_root))
        self.addCleanup(sys.path.remove, str(launch_root))
        spec = importlib.util.spec_from_file_location(
            "package_test_launcher", launch_root / "rollout_launch.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        source = export(self.root / "source", "smolvla")
        archive_path = self.root / "package.tar"
        archive(source, archive_path)
        directories = []

        @contextmanager
        def selected(task, checkpoint, mode, steps):
            directories.append(checkpoint)
            self.assertTrue((checkpoint / "model.safetensors").exists())
            yield task

        def submitted(args, task):
            self.assertTrue((directories[0] / "model.safetensors").exists())
            return 0

        cwd = Path.cwd()
        try:
            with (
                patch.object(
                    sys,
                    "argv",
                    ["rollout", "--checkpoint-archive", str(archive_path), "--validate-only"],
                ),
                patch.object(launcher, "_select_checkpoint", selected),
                patch.object(launcher, "_submit", submitted),
            ):
                self.assertEqual(launcher._main(), 0)
        finally:
            os.chdir(cwd)
        self.assertFalse(directories[0].exists())

    def test_pax_sparse_parser_rejected_before_reading_map(self):
        path = self.root / "sparse.tar"
        with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
            info = tarfile.TarInfo("sparse")
            info.size = 0
            info.pax_headers = {
                "GNU.sparse.major": "1",
                "GNU.sparse.minor": "0",
                "GNU.sparse.name": "sparse",
                "GNU.sparse.realsize": "999999999",
            }
            tar.addfile(info)
        with self.assertRaisesRegex(ValueError, "Sparse TAR files are unsupported"):
            with resolve_checkpoint(path, archive=True):
                self.fail("sparse parser entered")


if __name__ == "__main__":
    unittest.main()
