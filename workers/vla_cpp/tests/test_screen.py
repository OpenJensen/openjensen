import sys

import pytest

from policykit import screen


@pytest.mark.parametrize('run_id', ['smolvla-screen-v1', 'custom-screen'])
def test_existing_evidence_is_rejected_before_conversion_or_mutation(tmp_path, monkeypatch, run_id):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'artifacts/docker/runs' / run_id
    output.mkdir(parents=True)
    (output / 'results.json').write_text('{"results": ["original evidence"]}')
    (output / 'REPORT.md').write_text('Original report')
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    monkeypatch.setattr(sys, 'argv', ['screen'] + ([] if run_id == 'smolvla-screen-v1' else ['--run', run_id]))
    def fail(*args, **kwargs):
        raise AssertionError('Conversion would fail; existing evidence must be rejected first')
    monkeypatch.setattr(screen.subprocess, 'run', fail)
    monkeypatch.setattr(screen.subprocess, 'check_output', fail)
    with pytest.raises(SystemExit) as exc:
        screen.main()
    assert exc.value.code == 2
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


@pytest.mark.parametrize('run_id', ['..', '.', '../elsewhere', '/tmp/elsewhere', ''])
def test_run_id_cannot_escape_evidence_directory(tmp_path, monkeypatch, run_id):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['screen', '--run', run_id])
    with pytest.raises(SystemExit) as exc:
        screen.main()
    assert exc.value.code == 2
    assert not (tmp_path / 'artifacts').exists()
