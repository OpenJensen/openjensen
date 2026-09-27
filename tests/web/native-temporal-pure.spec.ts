import { test, expect } from '@playwright/test';
import { measuredNativeReport } from '../../apps/web/src/lib/native-quantization';
import { replayReport, readReplay, type ReplayJob } from '../../apps/web/src/lib/native-replay';
import type { Job } from '../../apps/web/src/lib/api';

const model = `sha256:${'a'.repeat(64)}`, sha = 'b'.repeat(64);
const quantJob = { id: 'packed', project_id: 'alpha', kind: 'policy.quantize', status: 'succeeded', request: { operation: 'policy.quantize', artifact_id: 'export', native_quantization: { format: 'firebird_quant', bits: 8, group_size: 64 } } } as Job;
function quantReport(prediction = 8, execution = 3) {
  return { stage: 'operation', operation: 'policy.quantize', architecture: 'act', format: 'firebird_quant', precision: 'int8', model_id: model, source_artifact_id: 'export', source_artifact_manifest_sha256: sha, fresh_reload_verified: true, cpu_reload_verified: true, quality_verified: false, calibration_verified: false, isaac_runtime_verified: false, runtime_verified: false, speedup_verified: false, task_success: null, gpu_memory_bytes: null, inference_speedup: null, source_weight_bytes: 2000, packed_weight_bytes: 1000, policy_package_bytes: 1500, prediction_horizon: prediction, execution_horizon: execution, temporal_contract_sha256: sha, drift_from_fp32: [171, 902].map(seed => ({ seed, input_sha256: sha, raw: { rmse: 0.001, maximum_absolute_difference: 0.002, coordinates: prediction * 6 }, postprocessed: { rmse: 0.01, maximum_absolute_difference: 0.02, coordinates: prediction * 6 } })) };
}
const replayJob = { id: 'run', project_id: 'alpha', kind: 'policy.run', status: 'succeeded', request: { operation: 'policy.run', artifact_id: 'packed', dataset_job_id: 'dataset', native_replay: { adapter: 'act-packed-observation-v1', coordinate_attestation: 'generated_fixture', selection: [{ episode_index: 2, frame_index: 3 }], units: ['degrees','degrees','degrees','degrees','degrees','recorded_gripper'] } } } as ReplayJob;
function replayEvidence(prediction = 8) {
  return { operation: 'policy.run', stage: 'native_replay', mode: 'independent_observation_replay', device: 'cpu', source_artifact_id: 'packed', dataset_job_id: 'dataset', observation_source: { kind: 'generated_fixture' }, model_id: model, observations: 1, action_shape: [prediction, 6], reset_repeat_exact: true, server_closed: true, task_success: null, quality_verified: false, calibration_verified: false, speedup_verified: false, isaac_runtime_verified: false, elapsed_seconds: 1 };
}
function record(prediction = 8) {
  return { artifact_id: 'run:operation', job_id: 'run', model_id: model, source_kind: 'generated_fixture', coordinate_names: ['j0','j1','j2','j3','j4','gripper'], units: replayJob.request.native_replay.units, records: [{ episode_index: 2, frame_index: 3, reset_repeat_exact: true, actions: Array.from({ length: prediction }, (_, i) => Array.from({ length: 6 }, (_, j) => i + j / 10)) }] };
}
const originalFetch = globalThis.fetch;
test.afterEach(() => { globalThis.fetch = originalFetch; });
test('8-prediction/3-execution quantization validates all48 drift coordinates', () => {
  expect(measuredNativeReport(quantReport(), quantJob)).not.toBeNull();
});
test('8-prediction replay accepts its complete8x6 record rather than execution3', async () => {
  const report = replayReport(replayEvidence(), replayJob); expect(report).not.toBeNull();
  globalThis.fetch = async () => Response.json(record());
  expect((await readReplay('alpha','run:operation',replayJob,report!)).records[0].actions).toHaveLength(8);
});

import { replayableNativeOutput, type PackedArtifact } from '../../apps/web/src/lib/native-quantization';
import { policyJobRequest } from '../../apps/web/src/lib/policy-job-mutation';

test('legacy100-step report and full record remain accepted with bounded defaults', async () => {
  const { prediction_horizon: _p, execution_horizon: _e, temporal_contract_sha256: _t, ...legacy } = quantReport(100,100);
  const measured = measuredNativeReport(legacy, quantJob)!; expect(measured.prediction_horizon).toBe(100); expect(measured.execution_horizon).toBe(100); expect(measured.temporal_contract_sha256).toBeNull();
  const report = replayReport(replayEvidence(100),replayJob)!; globalThis.fetch = async () => Response.json(record(100));
  expect((await readReplay('alpha','run:operation',replayJob,report)).records[0].actions).toHaveLength(100);
});
for (const [index, patch] of [{ prediction_horizon: true }, { prediction_horizon: 0 }, { prediction_horizon: 1025 }, { prediction_horizon: 8.5 }, { execution_horizon: true }, { execution_horizon: 0 }, { execution_horizon: 9 }, { execution_horizon: 1.5 }, { temporal_contract_sha256: [] }, { temporal_contract_sha256: 'bad' }, { execution_horizon: undefined }, { temporal_contract_sha256: undefined }].entries()) test(`strict bounded quantization dimensions reject malformed${index}`, () => {
  expect(measuredNativeReport({ ...quantReport(), ...patch },quantJob)).toBeNull();
});
for (const count of [18, 47, 49, 600, '48', true]) test(`quantization drift must cover all48 prediction coordinates: ${count}`, () => {
  const report = quantReport(); (report.drift_from_fp32[1].postprocessed as Record<string,unknown>).coordinates=count;
  expect(measuredNativeReport(report,quantJob)).toBeNull();
});
for (const claim of ['quality_verified','calibration_verified','runtime_verified','isaac_runtime_verified','speedup_verified']) test(`new dimensions cannot promote ${claim}`, () => {
  expect(measuredNativeReport({ ...quantReport(),[claim]:true },quantJob)).toBeNull();
  if(claim!=='runtime_verified') expect(replayReport({ ...replayEvidence(),[claim]:true },replayJob)).toBeNull();
});
test('temporal metadata is required to agree with the report before replay handoff', () => {
  const report=measuredNativeReport(quantReport(),quantJob)!;
  const metadata={ architecture:'act',format:'firebird_quant',format_version:1,model_id:model,precision:'int8',inference_only:true,fresh_reload_verified:true,cpu_reload_verified:true,source_artifact_id:'export',source_artifact_manifest_sha256:sha,prediction_horizon:8,execution_horizon:3,temporal_contract_sha256:sha };
  const artifact={id:'packed:policy',project_id:'alpha',job_id:quantJob.id,format:'native_quantized',metadata} as PackedArtifact;
  expect(replayableNativeOutput(artifact,quantJob,report)).toBe(true);
  for(const patch of [{prediction_horizon:100},{execution_horizon:8},{temporal_contract_sha256:'c'.repeat(64)}]) expect(replayableNativeOutput({...artifact,metadata:{...metadata,...patch}},quantJob,report)).toBe(false);
});
for(const shape of [[0,6],[1025,6],[8.5,6],[true,6],['8',6],[8,5],[8,6,1],null]) test(`replay rejects malformed shape${JSON.stringify(shape)}`,()=>{
  expect(replayReport({...replayEvidence(),action_shape:shape},replayJob)).toBeNull();
});
for(const length of [0,3,7,9,100]) test(`8-step report rejects incomplete/wrong actions length${length}`,async()=>{
  globalThis.fetch=async()=>Response.json(record(length));
  await expect(readReplay('alpha','run:operation',replayJob,replayReport(replayEvidence(),replayJob)!)).rejects.toThrow('complete finite 8 × 6');
});
for(const fault of ['dimensions','null','boolean','string','nonfinite','episode','frame','model','source','units','repeat']) test(`8-step records keep exact identity and finite checks:${fault}`,async()=>{
  const value=record();
  if(fault==='dimensions') value.records[0].actions[7].pop();
  else if(['null','boolean','string','nonfinite'].includes(fault)) (value.records[0].actions[7] as unknown[])[5]=({null:null,boolean:true,string:'1',nonfinite:Infinity} as Record<string,unknown>)[fault];
  else if(fault==='episode') value.records[0].episode_index=3;
  else if(fault==='frame') value.records[0].frame_index=4;
  else if(fault==='model') value.model_id=`sha256:${'c'.repeat(64)}`;
  else if(fault==='source') value.source_kind='lerobot_snapshot';
  else if(fault==='units') value.units=['other'];
  else (value.records[0] as Record<string,unknown>).reset_repeat_exact=false;
  globalThis.fetch=async()=>Response.json(value);
  await expect(readReplay('alpha','run:operation',replayJob,replayReport(replayEvidence(),replayJob)!)).rejects.toThrow();
});
for(const prediction of [1,1024]) test(`bounded boundary prediction${prediction} requires full arrays`,async()=>{
  expect(measuredNativeReport(quantReport(prediction,1),quantJob)).not.toBeNull();
  const report=replayReport(replayEvidence(prediction),replayJob)!; expect(report).not.toBeNull();
  globalThis.fetch=async()=>Response.json(record(prediction));
  expect((await readReplay('alpha','run:operation',replayJob,report)).records[0].actions).toHaveLength(prediction);
});
test('bounded replay read accepts full1024x6x32 record above default1MiB',async()=>{
  const selection=Array.from({length:32},(_,i)=>({episode_index:i,frame_index:0}));
  const ownJob={...replayJob,request:{...replayJob.request,native_replay:{...replayJob.request.native_replay,selection}}};
  const evidence={...replayEvidence(1024),observations:32};
  const value=record(1024);value.records=selection.map(row=>({...row,reset_repeat_exact:true,actions:Array.from({length:1024},(_,i)=>Array.from({length:6},(_,j)=>Math.PI+i+j/10))}));
  expect(new TextEncoder().encode(JSON.stringify(value)).length).toBeGreaterThan(1_048_576);
  globalThis.fetch=async()=>Response.json(value);
  expect((await readReplay('alpha','run:operation',ownJob,replayReport(evidence,ownJob)!)).records).toHaveLength(32);
});
for(const limit of [1_048_576,8_388_608] as const) test(`transport cancels and refuses response beyond${limit} bytes`,async()=>{
  let cancelled=false;
  globalThis.fetch=async()=>new Response(new ReadableStream({start(controller){controller.enqueue(new Uint8Array(limit+1));},cancel(){cancelled=true;}}));
  await expect(policyJobRequest('/bounded-read',undefined,limit===1_048_576?undefined:{maxResponseBytes:limit})).rejects.toThrow('too large');expect(cancelled).toBe(true);
});
test('large read option cannot raise mutation response bounds',async()=>{
  let calls=0;globalThis.fetch=async()=>{calls++;return Response.json({});};
  await expect(policyJobRequest('/mutation',{}, {maxResponseBytes:8_388_608})).rejects.toThrow('Invalid bounded response limit');expect(calls).toBe(0);
});
