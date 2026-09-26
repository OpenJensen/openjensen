"""Reproduce the recorded GGUF bytes from the pinned floating checkpoint."""
import json
from pathlib import Path
import subprocess
import sys

from policykit.worker import sha256 as file_hash
from policykit.quantization import quantize

root = Path.cwd()
source = root / 'artifacts/docker/sources/smolvla'
previous = root / 'artifacts/docker/runs/smolvla-screen-v1'
manifest = json.loads((previous / 'results.json').read_text())
if file_hash(source / 'model.safetensors') != manifest['source']['model_sha256']:
    raise RuntimeError('Source weight hash does not match the recorded checkpoint')
vendor = root / 'artifacts/docker/vendor/vla.cpp'
models = previous / 'models'
models.mkdir(exist_ok=True)
reference = models / 'smolvla-float.gguf'
verified = []
for row in manifest['results']:
    name = row['preset']
    model = models / Path(row['artifact']).name
    if model.exists():
        if file_hash(model) != row['sha256']:
            raise RuntimeError(f'Existing file has the wrong hash: {model}')
    else:
        temporary = model.with_suffix('.partial.gguf')
        if name == 'float_reference':
            subprocess.run([sys.executable, str(vendor / 'scripts/convert_smolvla_to_gguf.py'),
                            '--ckpt', str(source), '--out', str(temporary)], check=True)
        else:
            quantize(reference, temporary, vendor / 'scripts/quantize_gguf.py',
                     'Q4_0' if name.startswith('lm_q4') else 'Q8_0',
                     'Q8_0' if 'vision' in name else None)
        digest = file_hash(temporary)
        if digest != row['sha256']:
            raise RuntimeError(f'Reproduced {name} hash {digest} differs from {row["sha256"]}')
        temporary.replace(model)
        audit = temporary.with_suffix('.audit.json')
        if audit.exists():
            audit.replace(model.with_suffix('.audit.json'))
    verified.append({'preset': name, 'sha256': row['sha256'], 'path': str(model)})
    print(f'Verified identical artifact: {name}', flush=True)
(root / 'artifacts/cuda/setup/reproduced-models.json').write_text(json.dumps(verified, indent=2) + '\n')
