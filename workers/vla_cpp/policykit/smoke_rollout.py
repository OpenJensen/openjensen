"""Bounded, paired SmolVLA LIBERO smoke run; short runs never become task scores."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .benchmark import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', required=True)
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--action-steps', type=int, default=4)
    parser.add_argument('--port', type=int, default=5560)
    args = parser.parse_args()
    if not 1 <= args.steps <= 500 or not 1 <= args.action_steps <= 50:
        parser.error('Use 1..500 environment steps and 1..50 replayed actions')
    lane = Path.cwd()/'artifacts/docker'
    vendor = lane/'vendor/vla.cpp'
    manifest = json.loads((lane/'runs/smolvla-packed-v2/results.json').read_text())
    row = next((r for r in manifest['results'] if r['preset']==args.preset),None)
    if row is None or row['status'] != 'engine_verified':
        raise RuntimeError('Select an engine-verified candidate')
    model = Path(row['artifact'])
    if file_hash(model) != row['sha256']:
        raise RuntimeError('Model differs from the verified artifact')
    out = lane/'runs/smolvla-libero-smoke'/args.preset/f'task{args.task_id}-seed{args.seed}-steps{args.steps}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        raise RuntimeError('Preserve existing rollout evidence before repeating')
    result = {'preset':args.preset,'task':'libero_object','task_id':args.task_id,'seed':args.seed,
              'policy_noise_seed':args.seed,'action_steps':args.action_steps,'step_budget':args.steps,
              'artifact_sha256':row['sha256'],'status':'starting','task_success':None,'steps':0}
    def save():
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    server = env = client = None
    try:
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
        sys.path.insert(0,str(vendor/'eval'))
        import numpy as np
        import gymnasium as gym
        import sim.libero  # registers the upstream simulator adapter
        from client.vla_cpp_client import VlaCppClient
        from client.adapters import LeRobotPipelineAdapter
        result['simulator_commit'] = subprocess.check_output(
            ['git','-C',str(lane/'LIBERO'),'rev-parse','HEAD'],text=True).strip()
        env = gym.make(f'libero_object/task_{args.task_id}',seed=args.seed,
                       output_video_dir=out/'videos',video_fps=30,video_view_mode='multi-view')
        obs, _ = env.reset(seed=args.seed)
        result['status'] = 'environment_ready'
        save()
        with (out/'server.log').open('w') as server_log:
            command = [str(vendor/'build/vla-server'),'--bind',f'tcp://127.0.0.1:{args.port}',str(model)]
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
                                  n_action_steps=args.action_steps)
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
