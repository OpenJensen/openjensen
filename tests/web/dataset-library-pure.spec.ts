import { expect, test } from '@playwright/test';
import { datasetLibrary, datasetRequest, DatasetUploadError, labelDraftKey, readLabelDraft, sameLabelDraft, isProjectDataset, type DatasetRepairNotice } from '../../apps/web/src/lib/dataset-library';

const originalFetch = globalThis.fetch;
test.afterEach(() => { globalThis.fetch = originalFetch; });

test('dataset requests enforce a deadline and forward explicit cancellation', async () => {
  const signals: AbortSignal[] = [];
  globalThis.fetch = async (_url, init) => new Promise((_resolve, reject) => {
    const signal = init!.signal!;
    signals.push(signal);
    signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
  });
  await expect(datasetRequest('/datasets', undefined, 5)).rejects.toThrow('did not respond in time');
  expect(signals[0].aborted).toBe(true);
  const controller = new AbortController();
  const pending = datasetLibrary.list('project-one', controller.signal);
  controller.abort();
  await expect(pending).rejects.toThrow('cancelled');
  expect(signals[1].aborted).toBe(true);
  await expect(datasetRequest('/datasets', {signal: controller.signal})).rejects.toThrow('cancelled');
  expect(signals).toHaveLength(2);
});

test('upload failure retains its server identity and never retries or finishes', async () => {
  const calls: {url:string;method:string}[] = [];
  const id = 'a'.repeat(32);
  globalThis.fetch = async (url, init) => {
    calls.push({url:String(url),method:init?.method??'GET'});
    return calls.length === 1 ? new Response(JSON.stringify({id,project_id:'project-one',status:'uploading'})) : new Response(JSON.stringify({detail:'Connection interrupted'}),{status:503});
  };
  const started: string[] = [];
  const file = new File(['fixture'], 'records.csv');
  let failure: unknown;
  try { await datasetLibrary.upload('project-one',[file],()=>{},undefined,entry=>started.push(entry.id)); } catch (error) { failure=error; }
  expect(failure).toBeInstanceOf(DatasetUploadError);
  expect((failure as DatasetUploadError).uploadId).toBe(id);
  expect((failure as Error).message).toContain('Open the saved upload in My datasets');
  expect(started).toEqual([id]);
  expect(calls.map(call=>call.method)).toEqual(['POST','PUT']);
  expect(calls.some(call=>call.url.endsWith('/finish'))).toBe(false);
});

test('lost upload creation acknowledgement tells the user to inspect the collection', async () => {
  let calls = 0;
  globalThis.fetch = async () => {calls++;throw new TypeError('Connection lost');};
  await expect(datasetLibrary.upload('project-one',[new File(['fixture'],'records.csv')],()=>{})).rejects.toMatchObject({uploadId:null,message:expect.stringContaining('Refresh My datasets')});
  expect(calls).toBe(1);
});

test('stopping an acknowledged upload sends no later file or finish request',async()=>{
  const controller=new AbortController(),id='b'.repeat(32);let calls=0;
  globalThis.fetch=async()=>{calls++;return new Response(JSON.stringify({id,project_id:'project-one',status:'uploading'}));};
  await expect(datasetLibrary.upload('project-one',[new File(['fixture'],'records.csv')],()=>{},controller.signal,()=>controller.abort())).rejects.toMatchObject({uploadId:id,message:expect.stringContaining('cancelled')});
  expect(calls).toBe(1);
});

test('discard requests identify the exact upload and project', async () => {
  let seen = '';
  globalThis.fetch = async (url,init) => {seen=String(url);expect(init?.method).toBe('DELETE');expect(init?.cache).toBe('no-store');return new Response(JSON.stringify({discarded:true,id:'upload one'}));};
  expect(await datasetLibrary.discard('upload one','project/two')).toEqual({discarded:true,id:'upload one'});
  expect(seen).toContain('/dataset-uploads/upload%20one?project_id=project%2Ftwo');
});

test('label drafts retain their original revision and reject malformed browser data', () => {
  const labels = {views:{front:'Front view'},frames:{'0:1:front':'Approaching'}};
  expect(readLabelDraft(JSON.stringify({version:1,revision:4,labels}))).toEqual({revision:4,labels});
  for(const value of [null,'{',JSON.stringify({version:1,revision:-1,labels}),JSON.stringify({version:1,revision:1,labels:{views:[],frames:{}}}),JSON.stringify({version:1,revision:1,labels:{views:{front:42},frames:{}}})])expect(readLabelDraft(value)).toBeNull();
  expect(labelDraftKey({project_id:'one',id:'same'})).not.toBe(labelDraftKey({project_id:'two',id:'same'}));
  expect(labelDraftKey({project_id:'one',id:'first'})).not.toBe(labelDraftKey({project_id:'one',id:'second'}));
  expect(labelDraftKey({project_id:'one',id:'first'})).toMatch(/^jensen:dataset-labels:/);
  const accepted={revision:4,labels};
  expect(sameLabelDraft(readLabelDraft(JSON.stringify({version:1,...accepted})),accepted)).toBe(true);
  expect(sameLabelDraft({revision:4,labels:{...labels,views:{front:'Newer draft'}}},accepted)).toBe(false);
  expect(sameLabelDraft({revision:5,labels},accepted)).toBe(false);
});

test('unknown dataset ownership never enters project collections',()=>{
  const notice:DatasetRepairNotice={id:'a'.repeat(32),project_id:null,name:'Unavailable saved dataset',status:'unavailable',created_at:'',can_discard:false,error:'Record invalid; files preserved.'};
  expect(isProjectDataset(notice,'project-one')).toBe(false);
  expect(isProjectDataset(notice,'')).toBe(false);
});
