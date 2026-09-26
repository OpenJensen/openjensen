"""Freeze tokenizer/config files only; the full policy already includes VLM weights."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

repo = 'HuggingFaceTB/SmolVLM2-500M-Instruct'
out = Path('/workspace/artifacts/cuda/modelopt-metadata')
revision = '7b375e1b73b11138ff12fe22c8f2822d8fe03467'
snapshot_download(repo, revision=revision, local_dir=out,
                  allow_patterns=['*.json', '*.txt', '*.model', '*.jinja'])
files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(out.rglob('*')) if p.is_file() and '.cache' not in p.parts}
(out/'source.json').write_text(json.dumps({'repo_id':repo, 'revision':revision, 'files':files}, indent=2)+'\n')
print(out, revision, flush=True)
