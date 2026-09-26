"""Bounded, paired SmolVLA LIBERO smoke run; short runs never become task scores."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .worker import sha256 as file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', required=True)
    parser.add_argument('--lane', type=Path, help='Prepared simulator/vendor lane configured by the application')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init-state-id', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--action-steps', type=int, default=4)
    parser.add_argument('--port', type=int, default=5560)
    parser.add_argument('--rpc-timeout-ms', type=int, default=120_000)
    parser.add_argument('--run', default='smolvla-libero-smoke')
    parser.add_argument('--tokenizer')
    parser.add_argument('--capture-every', type=int, default=0)
    parser.add_argument('--max-captures', type=int, default=12)
    parser.add_argument('--manifest', type=Path, help='Engine-verified candidate manifest for this target')
    parser.add_argument('--runtime-build', type=Path, help='Native build containing vla-server')
    parser.add_argument('--output-root', type=Path, help='Parent directory for this target\'s rollout runs')
    parser.add_argument('--require-cuda', action='store_true', help='Reject policy-server CPU fallback')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in {'.', '..'}:
        parser.error('Run must be a single directory name')
    if args.capture_every < 0 or args.max_captures < 1:
        parser.error('Capture interval must be nonnegative and count positive')
    if args.rpc_timeout_ms <= 0:
        parser.error('RPC timeout must be positive')
    if args.init_state_id < 0:
        parser.error('Initial state index must be nonnegative')
    if not 1 <= args.steps <= 500 or not 1 <= args.action_steps <= 50:
        parser.error('Use 1..500 environment steps and 1..50 replayed actions')
    lane = args.lane or Path.cwd()/'artifacts/docker'
    vendor = lane/'vendor/vla.cpp'
    manifest_path = args.manifest or lane/'runs/smolvla-packed-v2/results.json'
    manifest = json.loads(manifest_path.read_text())
    server_binary = (args.runtime_build or vendor/'build')/'vla-server'
    row = next((r for r in manifest['results'] if r['preset']==args.preset),None)
    if row is None or row['status'] != 'engine_verified':
        raise RuntimeError('Select an engine-verified candidate')
    model = Path(row['artifact'])
    if file_hash(model) != row['sha256']:
        raise RuntimeError('Model differs from the verified artifact')
    out = (args.output_root or lane/'runs')/args.run/args.preset/f'task{args.task_id}-init{args.init_state_id}-seed{args.seed}-steps{args.steps}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        raise RuntimeError('Preserve existing rollout evidence before repeating')
    result = {'preset':args.preset,'task':'libero_object','task_id':args.task_id,'seed':args.seed,
              'evaluation_source':'LIBERO fixed benchmark initial states',
              'init_state_id':args.init_state_id,'training_overlap':'not audited',
              'rpc_timeout_ms':args.rpc_timeout_ms,'policy_noise_seed':args.seed,'action_steps':args.action_steps,'step_budget':args.steps,
              'artifact_sha256':row['sha256'],'runtime_binary_sha256':file_hash(server_binary),
              'engine_manifest':str(manifest_path),'engine_manifest_sha256':file_hash(manifest_path),
              'tokenizer':args.tokenizer,'capture_every':args.capture_every,'captures':[],'status':'starting','task_success':None,'steps':0}
    def save():
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    server = env = client = None
    try:
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        os.environ['LP_NUM_THREADS'] = '4'
        os.environ['MUJOCO_GL'] = 'osmesa'
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        config = lane/'libero-config'
        config.mkdir(exist_ok=True)
        base = lane/'LIBERO/libero/libero'
        import yaml
        (config/'config.yaml').write_text(yaml.safe_dump({
            'benchmark_root':str(base),'bddl_files':str(base/'bddl_files'),
            'init_states':str(base/'init_files'),'assets':str(base/'assets'),
            'datasets':str(lane/'LIBERO/libero/datasets')}))
        os.environ['LIBERO_CONFIG_PATH'] = str(config)
        sys.path.insert(0,str(lane/'LIBERO'))
        sys.path.insert(0,str(vendor/'eval'))
        import numpy as np
        import torch
        torch.set_num_threads(4)
        import gymnasium as gym
        import sim.libero  # registers the upstream simulator adapter
        from client.vla_cpp_client import VlaCppClient
        from client.adapters import LeRobotPipelineAdapter
        if (lane/'LIBERO/.git').exists():
            result['simulator_commit'] = subprocess.check_output(
                ['git','-C',str(lane/'LIBERO'),'rev-parse','HEAD'],text=True).strip()
        else:
            provenance = json.loads((lane/'LIBERO-source.json').read_text())
            archive = Path(provenance['archive'])
            if not archive.exists() and str(archive).startswith('/workspace/'):
                archive = lane.parent.parent / archive.relative_to('/workspace')
            if file_hash(archive) != provenance['archive_sha256']:
                raise RuntimeError('LIBERO source archive differs from its recorded hash')
            result['simulator_commit'] = provenance['revision']
            result['simulator_source'] = provenance
        env = gym.make(f'libero_object/task_{args.task_id}',seed=args.seed,
                       init_states=True,episode_index=args.init_state_id,
                       output_video_dir=out/'videos',video_fps=30,video_view_mode='multi-view')
        task = env.unwrapped.task_suite.tasks[args.task_id]
        state_path = base/'init_files'/task.problem_folder/task.init_states_file
        bddl_path = base/'bddl_files'/task.problem_folder/task.bddl_file
        if args.init_state_id >= len(env.unwrapped._init_states):
            raise ValueError('Initial state index exceeds benchmark states; refusing wraparound')
        result['evaluation_inputs'] = {
            'task_name':task.name, 'language':task.language,
            'initial_states':{'path':str(state_path),'sha256':file_hash(state_path)},
            'task_definition':{'path':str(bddl_path),'sha256':file_hash(bddl_path)},
        }
        obs, _ = env.reset(seed=args.seed)
        def space_mismatches(space, value, path='observation'):
            if hasattr(space, 'spaces') and isinstance(value, dict):
                return [item for key, child in space.spaces.items()
                        for item in space_mismatches(child, value.get(key), path+'.'+key)]
            return [] if space.contains(value) else [path]
        result['observation_space_mismatches'] = space_mismatches(env.observation_space, obs)
        result['status'] = 'environment_ready'
        save()
        with (out/'server.log').open('w') as server_log:
            command = [str(server_binary),'--bind',f'tcp://127.0.0.1:{args.port}',str(model)]
            result['server_command'] = command
            server = subprocess.Popen(command,stdout=server_log,stderr=subprocess.STDOUT,
                env={**os.environ,'VLA_N_THREADS':'4','OMP_NUM_THREADS':'4'})
            deadline = time.monotonic()+120
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Policy server exited during loading')
                try:
                    with socket.create_connection(('127.0.0.1',args.port),timeout=.2):
                        break
                except OSError:
                    if time.monotonic()>deadline:
                        raise TimeoutError('Policy server did not become ready')
                    time.sleep(.1)
            client = VlaCppClient(vla_addr=f'tcp://127.0.0.1:{args.port}',arch='smolvla',
                                  n_action_steps=args.action_steps,recv_timeout_ms=args.rpc_timeout_ms,
                                  tokenizer_name=args.tokenizer)
            server_text = (out/'server.log').read_text()
            result['cuda_required'] = args.require_cuda
            result['cuda_observed'] = 'backend = CUDA' in server_text and 'falling back to CPU' not in server_text
            if args.require_cuda and not result['cuda_observed']:
                raise RuntimeError('CUDA execution was not established for the policy server')
            # Couple policy noise across candidates as well as environment seeds.
            # The upstream SmolVLA server otherwise samples from random_device.
            original_socket = client.sock
            class SeededSocket:
                def __getattr__(self,name):
                    return getattr(original_socket,name)
                def send(self,body,*pos,**kw):
                    req = client.pb.PredictRequest()
                    req.ParseFromString(body)
                    del req.noise[:]
                    rng = np.random.default_rng(args.seed+req.request_id)
                    req.noise.extend(rng.standard_normal(50*32).astype(np.float32).tolist())
                    return original_socket.send(req.SerializeToString(),*pos,**kw)
            client.sock = SeededSocket()
            policy = LeRobotPipelineAdapter(client=client)
            policy.reset()
            result['status'] = 'running'
            save()
            for step in range(args.steps):
                if args.capture_every and step % args.capture_every == 0 and len(result['captures']) < args.max_captures:
                    parsed = policy.parse_observation(obs)
                    path = out/f'observation-{step:04d}.npz'
                    np.savez_compressed(path, **parsed)
                    result['captures'].append({'id':f'task{args.task_id}-init{args.init_state_id}-step{step}',
                        'step':step,'path':str(path),'sha256':file_hash(path),
                        'noise_seed':args.seed+step//args.action_steps})
                action = policy.get_action(obs)
                if not np.isfinite(action).all():
                    raise RuntimeError('Nonfinite robot action')
                obs, reward, terminated, truncated, info = env.step(action)
                result['steps'] = step+1
                if terminated or truncated:
                    result.update(status='episode_complete',task_success=bool(info.get('is_success',False)),
                                  terminated=bool(terminated),truncated=bool(truncated))
                    break
                if (step+1)%10 == 0:
                    print(f'{args.preset}: {step+1} environment steps',flush=True)
                    save()
            else:
                result['status'] = 'smoke_only'
                # Reaching an artificial step budget is not a task-failure score.
            save()
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        save()
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            try:
                if client is not None:
                    client.sock.close(linger=0)
            finally:
                if server is not None and server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
