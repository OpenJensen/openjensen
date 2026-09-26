"""Record explicit environment provenance on the remote Docker host (no credentials)."""
import json
from pathlib import Path
import subprocess

out = Path('artifacts/cuda/setup')
out.mkdir(parents=True, exist_ok=True)
def command(args):
    return subprocess.check_output(args, text=True).strip()

result = {
    'kernel': command(['uname', '-srmo']),
    'docker': command(['docker', '--version']),
    'gpu': command(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version,compute_cap', '--format=csv,noheader']),
    'images': [],
}
for name, python in [('firebird-quant-cuda', 'python3'), ('firebird-quant-sim', 'python3'), ('firebird-modelopt', 'python')]:
    tag = name+':20260926'
    data = json.loads(command(['docker', 'image', 'inspect', tag]))[0]
    result['images'].append({'tag':tag, 'id':data['Id'], 'created':data['Created'], 'architecture':data['Architecture']})
    (out/(name+'-packages.txt')).write_text(command(['docker','run','--rm','--entrypoint',python,tag,'-m','pip','freeze'])+'\n')
(out/'environment.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
