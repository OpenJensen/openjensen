"""Manual controls use the same bounded executor as voice tools."""

import argparse
import json
import os

from .contracts import OPERATIONS
from .control import Client
from .credentials import control_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=sorted(OPERATIONS | {"state"}))
    parser.add_argument(
        "--arguments", default="{}", help="JSON arguments; no Python/code execution"
    )
    args = parser.parse_args()
    client = Client(os.environ.get("FIREBIRD_TEACHING_CONTROL_URL", ""), control_token())
    result = (
        client.request("/state")
        if args.operation == "state"
        else client.command(args.operation, json.loads(args.arguments))
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
