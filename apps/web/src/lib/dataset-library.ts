import { apiOrigin, type DatasetProfile } from './api';

export type DatasetDetection = { format: string; convertible: boolean; reason?: string; sha256: string; file_count: number; total_bytes: number; episodes?: number; frames?: number; fps?: number; cameras: string[]; mapping: {action?: string; state?: string[]; cameras?: string[]; table?: string} };
export type DatasetLabels = { views: Record<string,string>; frames: Record<string,string> };
export type LibraryDataset = { id: string; project_id: string; name: string; source: 'huggingface' | 'local'; status: 'uploading' | 'detected' | 'converting' | 'ready' | 'failed'; created_at: string; job_id?: string; training_copy?: boolean; profile?: DatasetProfile; detection?: DatasetDetection; converted?: { format: string; episodes: number; frames: number; fps: number; cameras: string[] }; error?: string; preview_note?: string; example?: boolean; annotations?: DatasetLabels; annotation_revision?: number };
export type DatasetSample = { episode_index: number; frame_index: number; camera: string; path: string; task: string };
const base = `${apiOrigin}/api/v1`;
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(base + path, {...init, headers: {'Content-Type':'application/json',...init?.headers}});
  if (!response.ok) { const error = await response.json().catch(()=>({})); throw new Error(error.detail || `Dataset request failed (${response.status})`); }
  return response.json();
}
export const datasetLibrary = {
  list: (project?:string)=>request<LibraryDataset[]>(`/datasets${project ? `?project_id=${encodeURIComponent(project)}` : ''}`),
  get: (id:string)=>request<LibraryDataset>(`/datasets/${id}`),
  samples: (id:string)=>request<DatasetSample[]>(`/datasets/${id}/samples`),
  example: (project:string)=>request<LibraryDataset>(`/projects/${encodeURIComponent(project)}/datasets/example`,{method:'POST'}),
  convert: (id:string,settings:{fps:number;task:string;robot_type:string;mapping?:DatasetDetection['mapping']})=>request<LibraryDataset>(`/datasets/${id}/convert`,{method:'POST',body:JSON.stringify(settings)}),
  annotate: (id:string,revision:number,labels:DatasetLabels)=>request<LibraryDataset>(`/datasets/${id}/annotations`,{method:'PUT',body:JSON.stringify({revision,...labels})}),
  async upload(project:string, files:File[], onProgress:(value:string)=>void) {
    if (!files.length) throw new Error('Choose dataset files first.');
    if (files.length>4096 || files.reduce((sum,file)=>sum+file.size,0)>2*1024**3) throw new Error('Choose up to 4096 files and 2 GB.');
    const folder = files[0].webkitRelativePath.split('/')[0];
    const entry = await request<LibraryDataset>(`/projects/${encodeURIComponent(project)}/dataset-uploads`,{method:'POST',body:JSON.stringify({name:(folder || files[0].name).slice(0,100)})});
    for (let index=0;index<files.length;index++) {
      const file=files[index];
      const path=file.webkitRelativePath ? file.webkitRelativePath.split('/').slice(1).join('/') : file.name;
      onProgress(`Uploading ${index+1} of ${files.length} files…`);
      const response=await fetch(`${base}/dataset-uploads/${entry.id}/file?path=${encodeURIComponent(path)}`,{method:'PUT',headers:{'Content-Type':'application/octet-stream'},body:file});
      if (!response.ok) { const value=await response.json().catch(()=>({})); throw new Error(value.detail || 'Upload failed. Select the folder again to retry.'); }
    }
    onProgress('Detecting dataset format…');
    return request<LibraryDataset>(`/dataset-uploads/${entry.id}/finish`,{method:'POST'});
  },
};
export function sampleImage(id:string,name:string) { return `${base}/datasets/${id}/preview/${encodeURIComponent(name)}`; }
export function datasetDownload(id:string) { return `${base}/datasets/${id}/download`; }
export function formatName(value?:string) { return ({lerobot_v2:'LeRobot v2',lerobot_v3:'LeRobot v3',robomimic_hdf5:'robomimic HDF5',aloha_hdf5:'ALOHA HDF5',image_records:'Images + records',rlds_tfrecord:'RLDS / TFRecord',ros_recording:'ROS recording'} as Record<string,string>)[value??''] || 'Unknown format'; }
