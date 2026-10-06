import { test, expect } from '@playwright/test';
import { trainingMemory } from '../../apps/web/src/lib/training-memory';
import type { DatasetProfile } from '../../apps/web/src/lib/api';

const model = {id:'smolvla',label:'SmolVLA',description:'',model_id:'fixture/model',model_revision:'a'.repeat(40),methods:['lora','qlora'],suggested_gpu_memory_gb:16};
const camera = 'observation.images.front';
const profile = {features:{[camera]:{shape:[480,640,3]}},total_episodes:10,total_frames:1000} as unknown as DatasetProfile;
test('recipe estimates grow with simultaneous tensors, not dataset length or independent job count', () => {
 const base = trainingMemory(model,'qlora',4,[camera],profile)!;
 expect(trainingMemory(model,'qlora',64,[camera],profile)!.gb).toBeGreaterThan(base.gb);
 expect(trainingMemory(model,'lora',4,[camera],profile)!.gb).toBeGreaterThan(base.gb);
 expect(trainingMemory(model,'qlora',64,[camera,'second'],profile)!.gb).toBeGreaterThan(trainingMemory(model,'qlora',64,[camera],profile)!.gb);
 expect(trainingMemory(model,'qlora',4,[camera],{...profile,total_episodes:10000,total_frames:1000000})).toEqual(base);
 expect(trainingMemory(model,'qlora',4,[camera],profile,200)!.gb).toBeGreaterThan(base.gb);
 expect(trainingMemory({...model,id:'act',minimum_gpu_memory_gb:16},'full',4,[camera],profile,100)!.gb).toBeGreaterThanOrEqual(16);
 expect(trainingMemory({...model,id:'unknown',suggested_gpu_memory_gb:undefined},'full',4,[camera],profile)).toBeNull();
});
