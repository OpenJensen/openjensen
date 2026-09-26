"""Small ModelOpt numerical feasibility pilot; deliberately not a packed engine benchmark."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

from .worker import sha256 as file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['reference', 'awq', 'smoothquant'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--rollout', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = {'method': args.method, 'status': 'starting', 'task_success': None,
              'packed_engine': False, 'export_status': 'not_implemented_for_smolvla',
              'scope': 'Tiny within-trajectory numerical feasibility pilot; no task-quality or packed-runtime claim',
              'parameter_dtype': 'pending load audit', 'cuda_autocast_dtype': 'bfloat16',
              'rollout_sha256': file_hash(args.rollout)}
    def save():
        (args.out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    save()
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.nn import TensorQuantizer
        torch.set_num_threads(4)
        torch.manual_seed(42)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; CPU fallback is forbidden')
        result['versions'] = {p: importlib.metadata.version(p) for p in
                              ('torch', 'transformers', 'lerobot', 'nvidia-modelopt')}
        result['gpu'] = torch.cuda.get_device_name(0)
        source = Path('artifacts/docker/sources/smolvla')
        expected = '71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f'
        if file_hash(source/'model.safetensors') != expected:
            raise ValueError('Unexpected policy checkpoint')
        result['checkpoint_sha256'] = expected
        result['metadata'] = json.loads((args.metadata/'source.json').read_text())
        for relative, expected_hash in result['metadata']['files'].items():
            if file_hash(args.metadata/relative) != expected_hash:
                raise ValueError('Tokenizer/config metadata hash mismatch')
        cfg = PreTrainedConfig.from_pretrained(source)
        if not isinstance(cfg, SmolVLAConfig):
            raise ValueError('Expected a SmolVLA policy configuration')
        cfg.load_vlm_weights = False
        cfg.vlm_model_name = str(args.metadata.resolve())
        cfg.device = 'cuda'
        policy = SmolVLAPolicy.from_pretrained(source, config=cfg, strict=True).eval()
        dtype_counts = Counter()
        for parameter in policy.parameters():
            dtype_counts[str(parameter.dtype)] += parameter.numel()
        result['parameter_elements_by_dtype'] = dict(dtype_counts)
        result['parameter_dtype'] = ', '.join(sorted(dtype_counts))
        tokenizer = AutoTokenizer.from_pretrained(args.metadata, padding_side='right')
        stats = load_file(str(source/'policy_preprocessor_step_5_normalizer_processor.safetensors'))
        post_stats = load_file(str(source/'policy_postprocessor_step_1_unnormalizer_processor.safetensors'))
        result['normalization_keys'] = list(stats)
        def stat(values, feature, kind):
            matches = [v for k, v in values.items() if feature in k and k.endswith('.'+kind)]
            if len(matches) != 1:
                raise ValueError(f'Ambiguous or missing normalization statistic {feature}.{kind}')
            return matches[0].float().cuda()
        mean, std = stat(stats, 'observation.state', 'mean'), stat(stats, 'observation.state', 'std')
        action_mean, action_std = stat(post_stats, 'action', 'mean'), stat(post_stats, 'action', 'std')
        rollout = json.loads(args.rollout.read_text())
        cases = rollout['captures']
        if rollout['status'] != 'episode_complete' or len(cases) != 7:
            raise ValueError('Expected the frozen seven-observation completed development rollout')
        # Disjoint frames, but same trajectory: explicitly not an independent test set.
        calibration, evaluation = cases[:3], cases[3:]
        result['calibration'] = calibration
        result['evaluation'] = evaluation
        result['split_scope'] = '3 calibration frames, 4 disjoint frames from the same development trajectory'
        batches = {}
        for case in cases:
            path = Path(case['path'])
            if file_hash(path) != case['sha256']:
                raise ValueError('Captured observation hash mismatch')
            with np.load(path, allow_pickle=False) as data:
                batch = {key: torch.from_numpy(np.array(data[key], dtype=np.float32)).unsqueeze(0).cuda()
                         for key in ('observation.images.image', 'observation.images.image2', 'observation.state')}
                task = str(data['task'].item()).strip()+'\n'
            for key, value in batch.items():
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite captured input')
                if key.startswith('observation.images') and (value.shape[1] != 3 or value.min() < 0 or value.max() > 1):
                    raise ValueError('Expected CHW images in [0,1]')
            batch['observation.state'] = (batch['observation.state']-mean)/(std+1e-8)
            tokens = tokenizer([task], padding='longest', truncation=True, max_length=48, return_tensors='pt')
            batch['observation.language.tokens'] = tokens['input_ids'].cuda()
            batch['observation.language.attention_mask'] = tokens['attention_mask'].bool().cuda()
            noise = torch.from_numpy(np.random.default_rng(case['noise_seed']).standard_normal((1,50,32)).astype(np.float32)).cuda()
            batches[case['id']] = (batch, noise)

        prefix = 'model.vlm_with_expert.vlm.model.text_model.layers.'
        names = [name for name, module in policy.named_modules()
                 if name.startswith(prefix) and isinstance(module, torch.nn.Linear)]
        if len(names) != 224:
            raise ValueError(f'Expected 224 LM matrices, found {len(names)}')
        result['target_modules'] = names
        target_dtypes = Counter()
        for name, parameter in policy.named_parameters():
            if name.startswith(prefix):
                target_dtypes[str(parameter.dtype)] += parameter.numel()
        result['target_lm_parameter_elements_by_dtype'] = dict(target_dtypes)
        def protected_hash():
            digest = hashlib.sha256()
            for name, value in policy.named_parameters():
                if not name.startswith(prefix):
                    digest.update(name.encode())
                    digest.update(value.detach().float().cpu().numpy().tobytes())
            return digest.hexdigest()
        protected_before = protected_hash()
        def predict(model, case):
            batch, noise = batches[case['id']]
            model.reset()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                action = model.predict_action_chunk(batch, noise=noise.clone())
            if tuple(action.shape) != (1,50,7) or not torch.isfinite(action).all():
                raise ValueError('Expected finite 1 x 50 x 7 policy output')
            return (action.float()*action_std+action_mean).detach().cpu().numpy()
        # Establish the unchanged native model first, including all four evaluation frames.
        reference = {case['id']: predict(policy, case) for case in evaluation}
        np.savez_compressed(args.out/'reference-actions.npz', **reference)
        result['status'] = 'reference_verified'
        save()
        if args.method != 'reference':
            template = copy.deepcopy(mtq.INT4_AWQ_CFG if args.method == 'awq' else mtq.INT8_SMOOTHQUANT_CFG)
            quant_cfg = {'*': {'enable': False}}
            for name in names:
                quant_cfg[name+'.weight_quantizer'] = {**template['quant_cfg']['*weight_quantizer'], 'enable': True}
                if args.method == 'smoothquant':
                    quant_cfg[name+'.input_quantizer'] = {**template['quant_cfg']['*input_quantizer'], 'enable': True}
            quant_cfg['default'] = {'enable': False}
            config = {'quant_cfg': quant_cfg, 'algorithm': template['algorithm']}
            result['quantization_config'] = config
            save()
            def forward_loop(model):
                for case in calibration:
                    predict(model, case)
            started = time.monotonic()
            policy = mtq.quantize(policy, config, forward_loop)
            result['calibration_seconds'] = time.monotonic()-started
            enabled = [name for name, module in policy.named_modules()
                       if isinstance(module, TensorQuantizer) and module.is_enabled]
            expected_count = 224 if args.method == 'awq' else 448
            if len(enabled) != expected_count or any(not name.startswith(prefix) for name in enabled):
                raise ValueError(f'Quantizer scope mismatch: {len(enabled)} enabled; expected {expected_count}')
            result['enabled_quantizers'] = enabled
            result['protected_parameters_sha256_before'] = protected_before
            result['protected_parameters_sha256_after'] = protected_hash()
            if result['protected_parameters_sha256_after'] != protected_before:
                raise ValueError('Protected parameters were changed')
            candidate = {case['id']: predict(policy, case) for case in evaluation}
            np.savez_compressed(args.out/'candidate-actions.npz', **candidate)
            differences = np.concatenate([abs(candidate[k]-reference[k]) for k in reference], axis=0)
            result['action_mae'] = float(differences.mean())
            result['action_max_abs'] = float(differences.max())
            result['per_channel_mae'] = differences.mean(axis=(0,1)).tolist()
            result['status'] = 'fake_quant_numerics_verified'
            save()
            # Test the generic exporter on the complete policy, never only its VLM.
            # A checkpoint export alone would still require a compatible engine/reload check.
            try:
                from modelopt.torch.export import export_hf_checkpoint
                export_hf_checkpoint(policy, dtype=torch.bfloat16, export_dir=args.out/'export')
                result['export_status'] = 'checkpoint_written_reload_and_engine_unverified'
            except Exception as exc:
                result['export_status'] = 'failed'
                result['export_error'] = f'{type(exc).__name__}: {exc}'
                (args.out/'export-error.txt').write_text(traceback.format_exc())
        result['torch_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        result['torch_peak_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
        result['memory_scope'] = 'PyTorch pilot peak including model load/calibration; not a packed-engine memory result'
        save()
        print(json.dumps({k: result.get(k) for k in ('method','status','action_mae','action_max_abs')}, indent=2), flush=True)
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
