"""One request per process, one bounded JSON response; no actuator access."""

import asyncio
import json
import sys

from .contracts import MAX_REQUEST_BYTES, ProviderError, parse_json
from .providers import run


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ProviderError("invalid_request", "Provider request exceeds the size limit.")
        result = asyncio.run(run(parse_json(raw)))
        print(json.dumps(result, allow_nan=False, separators=(",", ":")))
        return 0
    except ProviderError as error:
        result = {"status": "error", "code": error.code, "error": str(error)}
        if hasattr(error, "usage"):
            result["usage"] = error.usage
        print(json.dumps(result))
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception:
        print(
            json.dumps(
                {
                    "status": "error",
                    "code": "worker_failed",
                    "error": "Planning failed; no proposal was admitted.",
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
