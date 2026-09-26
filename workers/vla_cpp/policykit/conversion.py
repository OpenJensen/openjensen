"""Content-bound floating conversion cache used by the benchmark CLI."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import tempfile

from .benchmark import file_hash


def tree_identity(root: Path, *, python_only: bool = False) -> dict[str, str]:
    if not root.is_dir():
        raise RuntimeError(f"Missing conversion input directory: {root}")
    files = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in {'.git', '.cache', '__pycache__'} for part in relative.parts):
            continue
        if path.is_file() and (not python_only or path.suffix == '.py'):
            files[relative.as_posix()] = file_hash(path)
    if not files:
        raise RuntimeError(f"No conversion inputs found in {root}")
    return files


def conversion_identity(config, runtime, model_name: str, source: Path, destination: Path) -> dict:
    model = config.model(model_name)
    converter = runtime.source_dir / model['converter']
    if not converter.is_file():
        raise RuntimeError(f"Missing converter: {converter}")
    prepared = config.artifacts / 'prepared-lock.json'
    source_lock = None
    if prepared.exists():
        source_lock = json.loads(prepared.read_text()).get('models', {}).get(model_name)
        if source_lock is not None and any(source_lock.get(key) != model.get(field)
                                           for key, field in [('source', 'source'), ('requested_revision', 'revision')]):
            raise RuntimeError('Prepared checkpoint does not match the configured source/revision; run prepare again')
    dependencies = {}
    for name in ('torch', 'safetensors', 'gguf', 'numpy', 'transformers'):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {
        'schema_version': 1,
        'checkpoint_files': tree_identity(source),
        'source_lock': source_lock,
        'model_settings': model,
        'runtime_settings': config.data['runtime'],
        'converter_sha256': file_hash(converter),
        # Include local helper changes as well as the entrypoint itself.
        'converter_python_files': tree_identity(converter.parent, python_only=True),
        'command': runtime.converter_command(model['converter'], source, destination),
        'python': {'executable': sys.executable, 'version': sys.version},
        'dependencies': dependencies,
    }


def cache_matches(artifact: Path, provenance: Path, identity: dict) -> bool:
    try:
        record = json.loads(provenance.read_text())
        return (artifact.stat().st_size > 0 and record['status'] == 'complete'
                and record['identity'] == identity and record['sha256'] == file_hash(artifact))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def convert_atomic(config, runtime, model_name, source, artifact, identity, execute):
    """A failed/stale conversion cannot replace a previously published reference."""
    provenance = artifact.with_suffix('.conversion.json')
    with tempfile.TemporaryDirectory(prefix='.conversion-', dir=artifact.parent) as temporary:
        output = Path(temporary) / artifact.name
        execute(runtime.converter_command(config.model(model_name)['converter'], source, output),
                cwd=runtime.source_dir)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError('Converter did not produce a nonempty floating artifact')
        if conversion_identity(config, runtime, model_name, source, artifact) != identity:
            raise RuntimeError('Conversion inputs changed during conversion; refusing to publish')
        record = {'status': 'complete', 'identity': identity, 'sha256': file_hash(output)}
        sidecar = Path(temporary) / provenance.name
        sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        output.replace(artifact)
        sidecar.replace(provenance)
