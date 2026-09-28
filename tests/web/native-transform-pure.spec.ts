import { expect, test } from '@playwright/test';
import type { DatasetJob, PolicyArtifact } from '../../apps/web/src/lib/api';
import { studentDatasetIssue, studentTeacher } from '../../apps/web/src/lib/native-distillation';
import { nativeQuantizationInput, nativeTransformMetadata } from '../../apps/web/src/lib/native-quantization';

// Generated metadata only: server file/hash validation is outside this UI contract.
function policy(metadata: Record<string, unknown> = {}): PolicyArtifact {
  return { id: 'teacher', project_id: 'alpha', job_id: 'export', format: 'inference_export', metadata: { architecture: 'act', inference_only: true, use_vae: false, ...metadata } } as PolicyArtifact;
}
function control() {
  return { schema_version: 1, kind: 'simulator_joint_position', controller: 'joint_position_targets', state_key: 'observation.state', action_key: 'action', state_units: 'radians', action_units: 'radians', timebase: 'simulation_seconds',
    joint_order: ['j0', 'j1', 'j2', 'j3', 'j4', 'gripper'], action_fps: 30,
    camera: { key: 'observation.images.front', width: 32, height: 32, prim: '/World/Camera' },
    source: { dataset_snapshot_id: `sha256:${'a'.repeat(64)}`, dataset_manifest_sha256: 'a'.repeat(64), demonstrations_sha256: 'b'.repeat(64), scene_sha256: ['c'.repeat(64)], scene_hash_scope: 'root USD bytes; referenced assets not inventoried', origins: ['synthetic'] },
    physical_calibration_verified: false, task_success_verified: false };
}
function claims() { return { control_contract: control(), control_contract_sha256: 'd'.repeat(64) }; }
const timing = { prediction_horizon: 32, execution_horizon: 8, temporal_contract_sha256: 'e'.repeat(64) };
function admitted(source: PolicyArtifact, expected: boolean) {
  expect(studentTeacher(source, 'alpha')).toBe(expected);
  expect(nativeQuantizationInput(source, 'alpha')).toBe(expected);
}

test('absent or null legacy control metadata retains genuine 100/100 default timing', () => {
  for (const metadata of [{}, { checkpoint: null, format: null }, { format: 'safetensors' }, { control_contract: null, control_contract_sha256: null }, { checkpoint: { control_contract: null, control_contract_sha256: null } }]) {
    const source = policy(metadata); admitted(source, true);
    expect(nativeTransformMetadata(source)).toMatchObject({ prediction_horizon: 100, execution_horizon: 100, temporal_contract_sha256: null, control_contract: null, control_contract_sha256: null });
  }
});
test('bounded independent horizons are supported at both metadata locations', () => {
  for (const prediction of [1, 32, 100, 1024]) for (const execution of [1, prediction]) {
    const dimensions = { prediction_horizon: prediction, execution_horizon: execution, temporal_contract_sha256: null };
    for (const metadata of [dimensions, { checkpoint: dimensions }, { ...dimensions, checkpoint: dimensions }]) {
      const source = policy(metadata); admitted(source, true);
      expect(nativeTransformMetadata(source)).toMatchObject(dimensions);
    }
  }
});
test('partial, malformed and contradictory timing does not silently become legacy timing', () => {
  for (const metadata of [{ prediction_horizon: 100 }, { execution_horizon: 1 }, { temporal_contract_sha256: null },
    ...[0, 1025, null, true, '32', [32]].map(prediction_horizon => ({ ...timing, prediction_horizon })),
    ...[0, 33, false, '8', [8]].map(execution_horizon => ({ ...timing, execution_horizon })),
    { ...timing, temporal_contract_sha256: 'invalid' }, { ...timing, checkpoint: { ...timing, execution_horizon: 4 } },
    { ...timing, checkpoint: { ...timing, temporal_contract_sha256: 'f'.repeat(64) } }]) admitted(policy(metadata), false);
});
test('imported saved config dimensions agree with any declared temporal metadata', () => {
  const imported = policy({ checkpoint: { chunk_size: 32, action_steps: 8, model_format: 'safetensors' } });
  admitted(imported, true); expect(nativeTransformMetadata(imported)).toMatchObject({ prediction_horizon: 32, execution_horizon: 8 });
  for (const checkpoint of [{ chunk_size: 32 }, { action_steps: 8 }, { chunk_size: true, action_steps: 1 }, { chunk_size: 32, action_steps: 33 }, { chunk_size: 64, action_steps: 8 }]) admitted(policy({ ...timing, checkpoint }), false);
});
test('paired simulator metadata supports both transform routes without inferring quality', () => {
  for (const metadata of [claims(), { checkpoint: claims() }, { ...claims(), checkpoint: claims() }, { control_contract: null, control_contract_sha256: null, checkpoint: claims() }]) {
    const source = policy({ ...metadata, ...timing }); admitted(source, true);
    expect(nativeTransformMetadata(source)?.control_contract).toEqual(control());
    expect(nativeTransformMetadata(source)?.control_contract?.physical_calibration_verified).toBe(false);
    expect(nativeTransformMetadata(source)?.control_contract?.task_success_verified).toBe(false);
  }
});
test('partial, malformed and contradictory simulator claims fail closed', () => {
  const changed = control(); changed.action_fps = 20;
  for (const metadata of [{ control_contract: control() }, { control_contract_sha256: 'd'.repeat(64) },
    { ...claims(), control_contract_sha256: 'invalid' }, ...[{}, [], '', 0, false].map(control_contract => ({ ...claims(), control_contract })),
    { ...claims(), checkpoint: { ...claims(), control_contract: changed } }, { ...claims(), checkpoint: { ...claims(), control_contract_sha256: 'f'.repeat(64) } },
    { ...claims(), control_contract: { ...control(), physical_calibration_verified: true } },
    { ...claims(), control_contract: { ...control(), joint_order: ['j0', 'j0', 'j2', 'j3', 'j4', 'gripper'] } },
    { ...claims(), control_contract: { ...control(), source: { ...control().source, dataset_snapshot_id: `sha256:${'f'.repeat(64)}` } } }]) admitted(policy(metadata), false);
});
test('packed imports stay outside FP32 transform routes and ownership and remote boundaries remain', () => {
  for (const metadata of [{ format: 'firebird_quant' }, { format: 'unknown' }, { format: [] }, { format: false }, { checkpoint: false }, { checkpoint: [] }, { checkpoint: 'bad' }, { checkpoint: { model_format: 'firebird_quant' } }, { format: 'safetensors', checkpoint: { model_format: 'firebird_quant' } }, { checkpoint: { model_format: null } }, { storage: 'gcs' }, { remote: true }, { remote_uri: 'gs://fixture/policy' }, { architecture: 'smolvla' }]) admitted(policy(metadata), false);
  expect(studentTeacher(policy(), 'beta')).toBe(false); expect(nativeQuantizationInput(policy(), 'beta')).toBe(false);
  admitted({ ...policy(), format: 'training_checkpoint' }, false);
  admitted({ ...policy(), format: 'native_quantized' }, false);
});
test('simulator distillation requires the exact source snapshot and explicit radians', () => {
  const teacher = policy(claims());
  const dataset = { project_id: 'alpha', result: { fps: 30, snapshot: { id: control().source.dataset_snapshot_id, manifest_sha256: control().source.dataset_manifest_sha256 } } } as DatasetJob;
  const units = Array<string>(6).fill('radians');
  expect(studentDatasetIssue(teacher, dataset, units)).toBeNull();
  expect(studentDatasetIssue(teacher, { ...dataset, project_id: 'beta' }, units)).toContain('original dataset snapshot');
  expect(studentDatasetIssue(teacher, { ...dataset, result: { ...dataset.result!, snapshot: { ...dataset.result!.snapshot!, id: `sha256:${'f'.repeat(64)}` } } }, units)).toContain('original dataset snapshot');
  expect(studentDatasetIssue(teacher, { ...dataset, result: { ...dataset.result!, fps: 20 } }, units)).toContain('cadence');
  expect(studentDatasetIssue(teacher, dataset, Array<string>(6).fill('degrees'))).toContain('radians');
  expect(studentDatasetIssue(policy(), dataset, Array<string>(6).fill('degrees'))).toBeNull();
});
