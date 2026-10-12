"""Quantize a local safetensors checkpoint without importing model code."""

import argparse
import json
from pathlib import Path

from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from .state import Recipe, load, quantize_state_dict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("quantize", help="Pack a floating safetensors state dictionary")
    convert.add_argument("source", type=Path)
    convert.add_argument("--out", required=True, type=Path)
    convert.add_argument("--bits", type=int, choices=(4, 8), default=4)
    convert.add_argument("--group-size", type=int, default=64)
    convert.add_argument("--min-elements", type=int, default=128)
    convert.add_argument("--min-ndim", type=int, default=2)
    convert.add_argument("--include", action="append", default=[])
    convert.add_argument("--exclude", action="append", default=[])
    inspect = commands.add_parser(
        "inspect", help="Validate an artifact and print its coverage audit"
    )
    inspect.add_argument("source", type=Path)
    restore = commands.add_parser("dequantize", help="Export ordinary floating safetensors weights")
    restore.add_argument("source", type=Path)
    restore.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "quantize":
            if args.out.exists():
                raise FileExistsError(args.out)
            state = quantize_state_dict(
                load_file(args.source),
                Recipe(
                    bits=args.bits,
                    group_size=args.group_size,
                    min_elements=args.min_elements,
                    min_ndim=args.min_ndim,
                    include=tuple(args.include),
                    exclude=tuple(args.exclude),
                ),
            )
            if not state.audit["quantized_elements"]:
                raise ValueError("No tensors qualify for compression; inspect the selection policy")
            state.save(args.out)
        else:
            state = load(args.source)
            if args.command == "dequantize":
                # Clone aliases: plain safetensors cannot represent shared storage.
                tensors = {name: tensor.clone() for name, tensor in state.dequantize().items()}
                import os
                import tempfile

                args.out.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix=".firebird-float-", dir=args.out.parent)
                os.close(fd)
                try:
                    save_file(tensors, temporary)
                    os.link(temporary, args.out)
                finally:
                    os.unlink(temporary)
        print(
            json.dumps(
                {
                    **state.audit,
                    "artifact_file_bytes": args.source.stat().st_size
                    if args.command == "inspect"
                    else args.out.stat().st_size,
                },
                indent=2,
            )
        )
    except (ValueError, OSError, KeyError, TypeError, SafetensorError) as error:
        parser.exit(2, f"firebird-quant: {error}\n")


if __name__ == "__main__":
    main()
