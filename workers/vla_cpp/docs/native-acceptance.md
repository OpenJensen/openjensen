# Run native worker checks

From the repository root, install the pinned development tools, then prepare
the worker's Python 3.11 environment:

```sh
uv sync --frozen
cd workers/vla_cpp
uv sync --locked --python 3.11 --extra quantize --extra test
```

Run the suite and source checks:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python -m pytest -q -rs -p no:cacheprovider
../../.venv/bin/ruff check policykit/acceptance.py policykit/provenance.py policykit/application.py \
  tests/test_application_acceptance.py tests/test_application.py
../../.venv/bin/ruff format --check policykit/acceptance.py policykit/provenance.py policykit/application.py \
  tests/test_application_acceptance.py tests/test_application.py
```

Prepare the pinned vendor source and Linux native runtime before running tests
that use them. Use the [runtime build recipe](../patches/README.md) and
[Spatial asset setup](spatial-application.md) for their input directories.
