"""Check CPU-only API imports and generated contracts without changing tracked files."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from vla_platform.api import create_app
from vla_platform.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
ML_MODULES = {"torch", "transformers", "lerobot", "openvla", "sky"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openapi", type=Path, default=ROOT / "packages/core/openapi.json")
    parser.add_argument("--client", type=Path, default=ROOT / "apps/web/src/lib/api.generated.ts")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="firebird-contracts-") as directory:
        temporary = Path(directory)
        schema = create_app(Settings(data_dir=temporary / "workspace")).openapi()
        imported = sorted({name.split(".")[0] for name in sys.modules} & ML_MODULES)
        if imported:
            print(f"FAIL: core imported ML/GPU modules: {', '.join(imported)}", file=sys.stderr)
            return 1
        print(
            "PASS: API schema imports no torch, transformers, lerobot, openvla or sky.", flush=True
        )

        generated_schema = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        # read_text accepts both checkout line endings while preserving all other content.
        if args.openapi.read_text(encoding="utf-8") != generated_schema:
            print("FAIL: OpenAPI drift; run the contract generators.", file=sys.stderr)
            return 1

        pnpm = shutil.which("pnpm")
        if pnpm is None:
            print("FAIL: pnpm is required to verify generated client types.", file=sys.stderr)
            return 1
        schema_path = temporary / "openapi.json"
        client_path = temporary / "api.generated.ts"
        schema_path.write_text(generated_schema, encoding="utf-8", newline="\n")
        subprocess.run(
            [pnpm, "exec", "openapi-typescript", str(schema_path), "-o", str(client_path)],
            cwd=ROOT,
            check=True,
            timeout=120,
        )
        if args.client.read_text(encoding="utf-8") != client_path.read_text(encoding="utf-8"):
            print("FAIL: TypeScript client drift; run the contract generators.", file=sys.stderr)
            return 1

    print("PASS: OpenAPI and TypeScript client match generated contracts (LF/CRLF normalized).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
