import json

import pytest

from firebird_decision import __main__ as cli
from firebird_decision.contracts import LICENSE, canonical


def args(path):
    return ["--model-dir", "/private/no-model", "--request", str(path), "--accept-license", LICENSE]


def test_missing_license_does_not_start_child(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_owned", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(SystemExit) as caught:
        cli.main(["--model-dir", str(tmp_path), "--request", str(tmp_path / "r")])
    assert caught.value.code == 2


@pytest.mark.parametrize("timeout", ["nan", "inf", "0", "-1", "121"])
def test_invalid_deadline(tmp_path, timeout, capsys):
    assert cli.main(args(tmp_path / "request") + ["--timeout-seconds", timeout]) == 2
    assert "Timeout" in capsys.readouterr().err


def test_private_request_snapshot_and_receipt_binding(
    tmp_path, monkeypatch, request_data, response_data, capsys
):
    source = tmp_path / "request.json"
    source.write_bytes(canonical(request_data))
    seen = []

    def child(command, **kwargs):
        from pathlib import Path

        path = Path(command[command.index("--request") + 1])
        source.write_text("{}")
        assert json.loads(path.read_text()) == request_data
        seen.append(path)
        return canonical(response_data)

    monkeypatch.setattr(cli, "run_owned", child)
    assert cli.main(args(source)) == 0
    assert not seen[0].exists()
    assert json.loads(capsys.readouterr().out)["selected_id"] == "lost"
    source.write_bytes(canonical(request_data))
    response_data["revision"] = "main"
    assert cli.main(args(source)) == 2
    assert not capsys.readouterr().out
