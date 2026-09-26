"""Offline identity/layout preflight. This never certifies a runnable policy or GPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import re
import sys
from pathlib import Path

from .spatial_protocol import verify_assets
from .worker import canonical, code_hash, sha256

NOT_CHECKED = ["gpu", "runtime", "episodes", "authorization", "action_parity", "model_load"]


def expected_hash(path, expected, label):
    if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
        raise ValueError(label + " needs an explicit lowercase SHA256")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(label + " SHA256 mismatch")
    return actual


def preflight(bundle, bundle_sha256, gguf_path, gguf_sha256):
    """Read caller-selected local assets; leave all inputs untouched."""
    import gguf

    from .quantization import validate_master
    from .spatial_fixture import load_arrays
    from .spatial_validation import validate_spatial_layout

    bundle, model = Path(bundle), Path(gguf_path)
    manifest = bundle / "spatial-assets.json"
    expected_hash(manifest, bundle_sha256, "Bundle manifest")
    expected_hash(model, gguf_sha256, "Floating GGUF")
    assets = verify_assets(bundle, require_reference=True)
    if assets["floating_gguf_sha256"] != gguf_sha256:
        raise ValueError("Floating GGUF differs from the prepared bundle binding")
    reader = gguf.GGUFReader(model)
    validate_master(reader)
    layout = validate_spatial_layout(reader, bundle, files=assets["files"])
    fixtures = []
    for name in assets["fixtures"]:
        arrays = load_arrays(bundle / name, expected_sha256=assets["files"][name])
        fixtures.append(
            {
                "path": name,
                "sha256": assets["files"][name],
                "arrays": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in sorted(arrays.items())
                },
            }
        )
    # Detect changes during static decoding; a receipt describes the verified bytes.
    if verify_assets(bundle, require_reference=True) != assets:
        raise ValueError("Spatial assets changed during preflight")
    expected_hash(manifest, bundle_sha256, "Bundle manifest")
    expected_hash(model, gguf_sha256, "Floating GGUF")
    lock = Path(__file__).resolve().parents[1] / "uv.lock"
    return {
        "schema_version": 1,
        "kind": "spatial_static_preflight",
        "status": "static_assets_verified",
        "scope": (
            "Hashes, declared GGUF layout, exact normalization and bounded fixture arrays only; "
            "not complete model weights, executable loading, policy quality or deployment readiness"
        ),
        "not_checked": NOT_CHECKED,
        "checks": {
            "bundle_inventory": "passed",
            "floating_gguf_identity": "passed",
            "floating_precision_layout_normalization": "passed",
            "fixture_arrays": "passed",
        },
        "inputs": {
            "bundle_path": str(bundle.resolve()),
            "spatial_assets_sha256": bundle_sha256,
            "gguf_path": str(model.resolve()),
            "gguf_sha256": gguf_sha256,
            "checkpoint": assets["checkpoint"],
            "backbone": assets["backbone"],
            "files": assets["files"],
        },
        "layout": layout,
        "fixtures": fixtures,
        "validator": {
            "worker_sha256": code_hash(),
            "python": platform.python_version(),
            "dependencies": {
                name: importlib.metadata.version(name) for name in ("gguf", "numpy", "safetensors")
            },
            "worker_lock_sha256": sha256(lock) if lock.is_file() else None,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--gguf-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = preflight(args.bundle, args.bundle_sha256, args.gguf, args.gguf_sha256)
    except Exception as exc:
        print(
            canonical(
                {
                    "schema_version": 1,
                    "kind": "spatial_static_preflight",
                    "status": "failed",
                    "not_checked": NOT_CHECKED,
                    "error": {"type": type(exc).__name__, "message": str(exc)[:2000]},
                }
            )
        )
        return 1
    print(canonical(receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
