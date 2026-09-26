"""Validate and summarize the five paired CUDA development episodes."""
import json
from pathlib import Path
from policykit.worker import sha256 as file_hash

root = Path('artifacts/cuda/runs/smolvla-cuda-libero-v1')
engine = Path('artifacts/cuda/runs/smolvla-cuda-v1/results.json')
manifest = json.loads(engine.read_text())
rows = []
identity = None
for candidate in manifest['results']:
    name = candidate['preset']
    path = root/name/'task0-init0-seed42-steps500/result.json'
    result = json.loads(path.read_text())
    assert result['status'] == 'episode_complete', (name, result['status'])
    assert result['cuda_observed'] and result['cuda_required']
    assert result['artifact_sha256'] == candidate['sha256']
    assert result['engine_manifest_sha256'] == file_hash(engine)
    current = {key: result[key] for key in ('task','task_id','init_state_id','seed',
               'policy_noise_seed','action_steps','step_budget','simulator_commit',
               'evaluation_inputs','runtime_binary_sha256','tokenizer')}
    if identity is None:
        identity = current
    assert current == identity, 'Paired protocol changed'
    videos = [{'path':str(p), 'sha256':file_hash(p), 'bytes':p.stat().st_size}
              for p in sorted(path.parent.rglob('*.mp4'))]
    assert videos and all(v['bytes'] > 0 for v in videos), f'Missing video for {name}'
    rows.append({'preset':name, 'task_success':result['task_success'], 'steps':result['steps'],
                 'terminated':result['terminated'], 'truncated':result['truncated'],
                 'artifact_sha256':result['artifact_sha256'], 'result_sha256':file_hash(path),
                 'observation_space_mismatches':result['observation_space_mismatches'], 'videos':videos})
payload = {'scope':'One paired development episode per candidate; no population success-rate claim',
           'protocol':identity, 'results':rows, 'deployment_winner':None}
(root/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
lines = ['# RTX 3070 paired LIBERO development pilot', '', payload['scope']+'.', '',
         '| Candidate | Success | Environment steps |', '|---|---|---:|']
lines += [f"| {r['preset']} | {r['task_success']} | {r['steps']} |" for r in rows]
lines += ['', 'Task 0 / initial state 0 / seed 42, 500-step natural horizon, four replayed actions per call. '
          'CUDA execution, candidate hashes, common protocol and nonempty videos were verified. '
          'Finite-action checks run on every environment step. The only observation-space warning is '
          'the upstream task-description field.', '',
          'This state was already used in CPU development. Broader paired episodes, native-policy/runtime '
          'parity and fresh-process deployment-package validation remain open. No deployment winner is selected.']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
