import { expect, test } from '@playwright/test';
import type { PolicyArtifact } from '../../apps/web/src/lib/api';
import { quantizationMemory } from '../../apps/web/src/lib/quantization-memory';

function artifact(format = 'training_checkpoint', metadata: Record<string, unknown> = {}): PolicyArtifact {
  return {id:'checkpoint',project_id:'project',job_id:'run',label:'Saved model',format:format as PolicyArtifact['format'],path:'owned',manifest_sha256:'a'.repeat(64),file_bytes:32,parent_ids:[],metadata:{architecture:'smolvla',base_model:{repository:'lerobot/smolvla_base'},...metadata}};
}

test('adapter bytes do not stand in for the full base and estimates describe job peaks',()=>{
  const small=artifact(),large={...small,file_bytes:20*1024**3};
  expect(quantizationMemory(small)).toMatchObject({gpuGiB:4,ramGiB:9});
  expect(quantizationMemory(large)).toEqual(quantizationMemory(small));
  expect(quantizationMemory(small)?.explanation).toContain('does not shrink with the selected output precision');
});
test('CPU packing and GPU checkpoint export have separate memory requirements',()=>{
  expect(quantizationMemory(artifact('native_checkpoint'))).toMatchObject({gpuGiB:0,ramGiB:9});
  expect(quantizationMemory(artifact('native_checkpoint'),true)).toMatchObject({gpuGiB:4,ramGiB:9});
});
test('unknown model identities and invalid counts cannot produce invented estimates',()=>{
  expect(quantizationMemory(artifact('training_checkpoint',{base_model:{repository:'custom/model'}}))).toBeNull();
  expect(quantizationMemory(artifact('training_checkpoint',{architecture:'act'}))).toBeNull();
  expect(quantizationMemory(artifact('training_checkpoint',{base_model:null,parameter_count:NaN}))).toBeNull();
  expect(quantizationMemory(artifact('training_checkpoint',{base_model:null,parameter_count:2**60}))).toBeNull();
});
test('recorded parameter counts and floating weight bytes override catalog assumptions',()=>{
  expect(quantizationMemory(artifact('training_checkpoint',{parameter_count:900_000_000}))).toMatchObject({gpuGiB:6,ramGiB:16});
  const gguf=artifact('gguf',{base_model:null,precision:'float',weight_bytes:1024**3});
  expect(quantizationMemory(gguf)).toMatchObject({gpuGiB:0,ramGiB:10});
  expect(quantizationMemory({...gguf,metadata:{...gguf.metadata,precision:'Q8_0'}})).toBeNull();
});
