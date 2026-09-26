"""Inspect dataset/checkpoint compatibility before running a VLA evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def assess(info: dict, model: dict, profile: dict) -> dict:
    features = info['features']
    inputs, outputs = model['input_features'], model['output_features']
    dataset_cameras = sorted(k for k,v in features.items() if v.get('dtype') in {'video','image'})
    model_cameras = sorted(k for k,v in inputs.items() if v.get('type') == 'VISUAL')
    mismatches = []
    for key, model_features in [('observation.state',inputs),('action',outputs)]:
        actual, expected = features[key]['shape'], model_features[key]['shape']
        if actual != expected:
            mismatches.append(f'{key}: dataset shape {actual}, checkpoint expects {expected}')
    if dataset_cameras != model_cameras:
        mismatches.append(f'Camera mapping required: dataset {dataset_cameras}, checkpoint {model_cameras}')
    contract = profile['dataset']
    if features['action']['shape'] != [contract['expected_action_dim']] or features['observation.state']['shape'] != [contract['expected_state_dim']]:
        raise ValueError('Downloaded data does not match the registered SO-101 feature dimensions')
    if contract['camera'] not in dataset_cameras:
        raise ValueError('Registered dataset camera is absent')
    requirements = ['A compatible SO-101 checkpoint and its saved normalization/action semantics',
                    'Disjoint evaluation episodes from the checkpoint training split',
                    'A matching simulator scene, controller, calibrated cameras and reset distribution',
                    'A task-success predicate evaluated on policy-controlled rollouts']
    return {
        'dataset':contract['repository'], 'revision':contract['revision'],
        'episodes':info['total_episodes'], 'frames':info['total_frames'], 'fps':info['fps'],
        'dataset_action_names':features['action'].get('names'),
        'dataset_cameras':dataset_cameras, 'checkpoint_cameras':model_cameras,
        'feature_mismatches':mismatches,
        'offline_evaluation_status':'blocked_incompatible_checkpoint' if mismatches else 'requires_checkpoint_semantics_and_split_validation',
        'closed_loop_status':'not_configured' if not profile['evaluation'].get('simulator') else 'requires_environment_validation',
        'task_success_drop_pp':None,
        'reason':'Recorded demonstrations do not define an executable simulator or measure this policy succeeding.',
        'required_for_task_success':requirements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,default=Path('configs/evaluation.so101.yaml'))
    parser.add_argument('--model-config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    profile=yaml.safe_load(args.profile.read_text())
    root=args.profile.resolve().parent.parent
    dataset_root=root/profile['dataset']['root']
    manifest=json.loads((dataset_root/'download-manifest.json').read_text())
    if manifest.get('revision') != profile['dataset']['revision'] or manifest.get('repository') != profile['dataset']['repository']:
        parser.error('Cached dataset provenance differs from the pinned profile')
    import hashlib
    info_path=dataset_root/'meta/info.json'
    if hashlib.sha256(info_path.read_bytes()).hexdigest() != manifest['files']['meta/info.json']['sha256']:
        parser.error('Cached dataset metadata hash mismatch')
    result=assess(json.loads(info_path.read_text()),json.loads(args.model_config.read_text()),profile)
    result['model_config']=str(args.model_config.resolve())
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# SO-101 pickup evaluation preflight','',f"Dataset: `{result['dataset']}` at `{result['revision']}`",'',
           f"{result['episodes']} episodes, {result['frames']} frames, {result['fps']} FPS.",'',
           f"**Offline evaluation: {result['offline_evaluation_status']}**",'',
           *['- '+m for m in result['feature_mismatches']],'',
           '**Task-success drop: unmeasured.** '+result['reason'],'',
           'Required before reporting task success:','',*['- '+r for r in result['required_for_task_success']],'']
    (args.out/'REPORT.md').write_text('\n'.join(lines))
    print(args.out/'REPORT.md')


if __name__ == '__main__':
    main()
