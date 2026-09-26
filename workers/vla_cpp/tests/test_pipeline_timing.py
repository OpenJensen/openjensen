import json
from pathlib import Path

import pytest

from policykit import cli, benchmark
from policykit.benchmark import BenchmarkRunner, Result, artifact_path, file_hash
from policykit.config import BenchmarkConfig


@pytest.fixture
def config(tmp_path):
    return BenchmarkConfig(tmp_path/'configs'/'test.yaml', {
        'runtime': {'source_dir':'vendor','build_dir':'build','cache_dir':'cache','tag':'test'},
        'models': {'smolvla': {'converter':'scripts/convert.py', 'source':'test/source', 'revision':'main'}},
        'presets': {'bf16':None,'q4_0':'Q4_0'},
        'evaluation': {'benchmark_repetitions':20}, 'machine_label':'test'})


def setup_cli(monkeypatch, config):
    source = config.artifacts / 'sources' / 'smolvla'
    source.mkdir(parents=True)
    (source / 'model.safetensors').write_bytes(b'source A')
    scripts = config.root / 'vendor' / 'scripts'
    scripts.mkdir(parents=True)
    (scripts / 'convert.py').write_text('# converter')
    monkeypatch.setattr(cli, 'load_config', lambda _: config)
    monkeypatch.setattr(cli.VlaCpp, 'require_prepared', lambda _: None)


def test_quantization_persists_separate_conversion_and_process_times(monkeypatch, config):
    setup_cli(monkeypatch, config)
    def run(command, **kwargs):
        Path(command[command.index('--out')+1]).write_bytes(b'weights')
    monkeypatch.setattr(cli, 'run', run)
    cli.quantize('unused', 'smolvla', 'q4_0')
    path = artifact_path(config, 'smolvla', 'q4_0')
    record = json.loads(path.with_suffix('.timing.json').read_text())
    assert record['status'] == 'complete'
    assert record['sha256'] == file_hash(path)
    assert record['conversion_reused'] is False
    assert set(record['timings']) == {'conversion_seconds','quantize_seconds','module_wall_seconds','artifact_hash_seconds','conversion_cache_check_seconds'}
    assert record['timings']['module_wall_seconds'] >= sum(record['timings'][k] for k in ('conversion_seconds','quantize_seconds','artifact_hash_seconds','conversion_cache_check_seconds'))
    cli.quantize('unused', 'smolvla', 'q4_0')
    record = json.loads(path.with_suffix('.timing.json').read_text())
    assert record['conversion_reused'] is True
    assert record['timings']['conversion_seconds'] == 0


def test_failed_quantization_saves_elapsed_not_a_success(monkeypatch, config):
    setup_cli(monkeypatch, config)
    monkeypatch.setattr(cli, 'run', lambda command, **kw: Path(command[command.index('--out')+1]).write_bytes(b'float'))
    cli.quantize('unused', 'smolvla', 'bf16')
    def fail(*args, **kwargs):
        raise RuntimeError('quantizer failed')
    monkeypatch.setattr(cli, 'run', fail)
    with pytest.raises(RuntimeError, match='quantizer failed'):
        cli.quantize('unused','smolvla','q4_0')
    record = json.loads(artifact_path(config,'smolvla','q4_0').with_suffix('.timing.json').read_text())
    assert record['status'] == 'failed'
    assert record['timings']['quantize_seconds'] >= 0
    assert record['timings']['module_wall_seconds'] >= record['timings']['quantize_seconds']
    assert 'sha256' not in record


def test_engine_only_uses_timings_for_matching_artifact(monkeypatch, config):
    path = artifact_path(config, 'smolvla', 'q4_0')
    path.parent.mkdir(parents=True)
    path.write_bytes(b'weights')
    sidecar = {'status':'complete','sha256':file_hash(path),
               'timings':{'quantize_seconds':5,'conversion_seconds':3,'module_wall_seconds':9}}
    path.with_suffix('.timing.json').write_text(json.dumps(sidecar))
    monkeypatch.setattr(benchmark, 'run', lambda *a, **k: 'load_ms=25\n| smolvla | 2 | 512 | 16 | 90.0 | 100.0 | 110.0 | 20.0 |\n')
    runner = BenchmarkRunner(config,'test')
    result = Result('smolvla','q4_0')
    runner.collect_engine_metrics(result)
    assert result.status == 'engine_benchmarked'
    assert result.quantize_seconds == 5
    assert result.load_ms == 25
    assert len(result.engine_process_seconds) == 3
    assert len(list(runner.output_dir.glob('*-bench-*.md'))) == 3
    path.write_bytes(b'changed')
    fresh = Result('smolvla','q4_0')
    runner.collect_engine_metrics(fresh)
    assert fresh.quantize_seconds is None
    # Existing result objects must not retain timing from the old artifact either.
    runner.collect_engine_metrics(result)
    assert result.quantize_seconds is None


def test_missing_engine_latency_is_failure(monkeypatch, config):
    path = artifact_path(config,'smolvla','q4_0')
    path.parent.mkdir(parents=True)
    path.write_bytes(b'weights')
    monkeypatch.setattr(benchmark,'run',lambda *a, **k:'no measurements')
    result = Result('smolvla','q4_0')
    BenchmarkRunner(config,'test').collect_engine_metrics(result)
    assert result.status == 'failed'


@pytest.mark.parametrize('change', ['checkpoint', 'converter', 'helper', 'revision', 'missing_provenance', 'corrupt_provenance', 'damaged_artifact'])
def test_conversion_cache_rebuilds_on_changed_or_unproven_inputs(monkeypatch, config, change):
    setup_cli(monkeypatch, config)
    calls = []
    source = config.artifacts / 'sources/smolvla/model.safetensors'
    scripts = config.root / 'vendor/scripts'
    def execute(command, **kwargs):
        calls.append(command)
        Path(command[command.index('--out')+1]).write_bytes(source.read_bytes())
    monkeypatch.setattr(cli, 'run', execute)
    cli.quantize('unused', 'smolvla', 'bf16')
    reference = artifact_path(config, 'smolvla', 'bf16')
    provenance = reference.with_suffix('.conversion.json')
    if change == 'checkpoint':
        source.write_bytes(b'source B')
    elif change == 'converter':
        (scripts / 'convert.py').write_text('# new converter')
    elif change == 'helper':
        (scripts / 'helper.py').write_text('# new helper')
    elif change == 'revision':
        config.data['models']['smolvla']['revision'] = 'new-revision'
    elif change == 'missing_provenance':
        provenance.unlink()
    elif change == 'corrupt_provenance':
        provenance.write_text('{partial')
    else:
        reference.write_bytes(b'partial')
    cli.quantize('unused', 'smolvla', 'bf16')
    record = json.loads(reference.with_suffix('.timing.json').read_text())
    assert len(calls) == 2
    assert record['status'] == 'complete'
    assert record['conversion_reused'] is False
    assert reference.read_bytes() == source.read_bytes()
    assert json.loads(provenance.read_text())['sha256'] == file_hash(reference)


def test_failed_conversion_preserves_previous_reference_and_records_failure(monkeypatch, config):
    setup_cli(monkeypatch, config)
    monkeypatch.setattr(cli, 'run', lambda command, **kw: Path(command[command.index('--out')+1]).write_bytes(b'old valid weights'))
    cli.quantize('unused', 'smolvla', 'bf16')
    reference = artifact_path(config, 'smolvla', 'bf16')
    old_provenance = reference.with_suffix('.conversion.json').read_bytes()
    (config.artifacts / 'sources/smolvla/model.safetensors').write_bytes(b'source B')
    def fail(command, **kwargs):
        Path(command[command.index('--out')+1]).write_bytes(b'partial')
        raise RuntimeError('converter failed')
    monkeypatch.setattr(cli, 'run', fail)
    with pytest.raises(RuntimeError, match='converter failed'):
        cli.quantize('unused', 'smolvla', 'bf16')
    record = json.loads(reference.with_suffix('.timing.json').read_text())
    assert reference.read_bytes() == b'old valid weights'
    assert reference.with_suffix('.conversion.json').read_bytes() == old_provenance
    assert record['status'] == 'failed'
    assert record['conversion_reused'] is False
    assert record['timings']['conversion_seconds'] >= 0
    assert 'sha256' not in record
    assert not list(reference.parent.glob('.conversion-*'))


def test_prepared_source_must_match_configured_source(monkeypatch, config):
    setup_cli(monkeypatch, config)
    (config.artifacts / 'prepared-lock.json').write_text(json.dumps({'models': {
        'smolvla': {'source': 'old/source', 'requested_revision': 'main', 'resolved_revision': 'abc'}
    }}))
    def forbidden(*a, **kw):
        pytest.fail('must reject stale prepared input before conversion')
    monkeypatch.setattr(cli, 'run', forbidden)
    with pytest.raises(RuntimeError, match='run prepare again'):
        cli.quantize('unused', 'smolvla', 'bf16')
    record = json.loads(artifact_path(config, 'smolvla', 'bf16').with_suffix('.timing.json').read_text())
    assert record['status'] == 'failed'


def test_inputs_changed_during_conversion_are_not_published(monkeypatch, config):
    setup_cli(monkeypatch, config)
    def execute(command, **kwargs):
        Path(command[command.index('--out')+1]).write_bytes(b'weights')
        (config.artifacts / 'sources/smolvla/model.safetensors').write_bytes(b'changed')
    monkeypatch.setattr(cli, 'run', execute)
    with pytest.raises(RuntimeError, match='inputs changed'):
        cli.quantize('unused', 'smolvla', 'bf16')
    reference = artifact_path(config, 'smolvla', 'bf16')
    assert not reference.exists()
    assert not reference.with_suffix('.conversion.json').exists()


@pytest.mark.parametrize('output_kind', ['missing', 'empty'])
def test_incomplete_conversion_output_is_not_published(monkeypatch, config, output_kind):
    setup_cli(monkeypatch, config)
    def execute(command, **kwargs):
        if output_kind == 'empty':
            Path(command[command.index('--out')+1]).touch()
    monkeypatch.setattr(cli, 'run', execute)
    with pytest.raises(RuntimeError, match='nonempty floating artifact'):
        cli.quantize('unused', 'smolvla', 'bf16')
    reference = artifact_path(config, 'smolvla', 'bf16')
    assert not reference.exists()
    assert not reference.with_suffix('.conversion.json').exists()
    assert json.loads(reference.with_suffix('.timing.json').read_text())['status'] == 'failed'
