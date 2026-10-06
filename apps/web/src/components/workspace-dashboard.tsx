'use client';

import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type Project } from '@/lib/api';
import { datasetLibrary, type LibraryDataset } from '@/lib/dataset-library';
import { ownedModels } from '@/lib/model-library';
import { Icon } from './icon';
import './workspace-dashboard.css';

export function WorkspaceDashboard({projects,onModels,onDatasets,onDataset,onSettings}:{projects:Project[];onModels:()=>void;onDatasets:()=>void;onDataset:(entry:LibraryDataset)=>void;onSettings:()=>void}) {
  const client=useQueryClient();
  const datasets=useQuery({queryKey:['datasets','all'],queryFn:()=>datasetLibrary.list(),retry:false,refetchInterval:10000});
  const models=useQueries({queries:projects.map(project=>({queryKey:['artifacts',project.id],queryFn:()=>api.artifacts(project.id),retry:false,refetchInterval:10000}))});
  const compute=useQuery({queryKey:['compute-settings'],queryFn:api.computeSettings,retry:false,refetchInterval:15000});
  const setup=useQuery({queryKey:['local-setup-status'],queryFn:api.localSetupStatus,retry:false});
  const hf=useQuery({queryKey:['huggingface-connection'],queryFn:api.huggingFaceStatus,retry:false});
  const discovery=useQuery({queryKey:['dashboard-local-discovery'],queryFn:api.checkLocalWorkers,enabled:false,retry:false});
  const check=useMutation({mutationFn:async()=>{const results=await Promise.allSettled([api.checkCloudCompute(),discovery.refetch()]);await client.invalidateQueries({queryKey:['compute-settings']});await setup.refetch();const failed=results.find(result=>result.status==='rejected');if(failed?.status==='rejected')throw failed.reason;}});
  const modelCount=models.reduce((sum,result,index)=>sum+ownedModels(result.data??[],projects[index].id).length,0);
  const modelsPending=models.some(result=>result.isPending),modelsError=models.some(result=>result.isError);
  const runtimes=compute.data?.runtimes.filter(runtime=>runtime.provider==='local')??[];
  const cloud=compute.data?.gcp_status;
  const candidate=discovery.data?.candidates[0];
  const localReady=candidate?.status==='registered' || runtimes.some(runtime=>runtime.training);
  const localName=candidate?.gpu_name??runtimes.find(runtime=>runtime.gpu_name)?.gpu_name;
  return <div className="workspace-dashboard">
    <section className="dashboard-welcome"><div><span className="dashboard-eyebrow">YOUR WORKSPACE</span><h2>Everything you’re working with.</h2><p>Models, datasets and the resources that run them.</p></div><span className="dashboard-projects">{projects.length} {projects.length===1?'project':'projects'}</span></section>
    <div className="dashboard-collections"><button className="dashboard-collection" aria-label="My models" onClick={onModels}><span className="dashboard-collection-icon"><Icon name="layers" size={25}/></span><div><h3>My models</h3><p>{modelsError?'Some models unavailable':modelsPending?'Loading models…':`${modelCount} saved models and versions`}</p></div><Icon name="arrow" size={20}/></button><button className="dashboard-collection" aria-label="My datasets" onClick={onDatasets}><span className="dashboard-collection-icon"><Icon name="database" size={25}/></span><div><h3>My datasets</h3><p>{datasets.isError?'Dataset history unavailable':datasets.isPending?'Loading datasets…':`${datasets.data?.length??0} saved datasets`}</p></div><Icon name="arrow" size={20}/></button></div>
    <section className="dashboard-resources" aria-labelledby="dashboard-resources-title"><div className="library-heading"><div><h2 id="dashboard-resources-title">Resources & connections</h2><p>Readiness is checked separately for each resource.</p></div><button className="secondary-button" disabled={check.isPending} onClick={()=>check.mutate()}>{check.isPending?'Checking resources…':'Check resources'}</button></div>
      {compute.isError && <p role="alert" className="error-notice">Resource settings unavailable. <button className="text-button" onClick={()=>void compute.refetch()}>Retry</button></p>}
      <div className="dashboard-resource-grid"><article><div className="dashboard-resource-heading"><Icon name="layers" size={20}/><h3>Local compute</h3><span className={`resource-state${localReady?' ready':''}`}>{compute.isPending?'Loading':localReady ? compute.data?.local.enabled?'Ready':'Disabled' : setup.data?.status==='succeeded'?'Installed':'Setup needed'}</span></div><p>{localName??compute.data?.local.label??'App host'}</p><small>{localReady?'SmolVLA training environment registered':candidate?.reason??(setup.data?.status==='succeeded'?'Training environment installed; check this machine to register it.':'Check this machine or install the training environment.')}</small><button className="text-link" onClick={onSettings}>{compute.data?.local.enabled?'Manage local runs':'Configure local runs'} →</button></article>
      <article><div className="dashboard-resource-heading"><Icon name="sliders" size={20}/><h3>Google Cloud</h3><span className={`resource-state${cloud?.status==='ready'?' ready':''}`}>{compute.isPending?'Loading':cloud?.status==='ready'?'Ready':cloud?.configured?'Setup needed':'Not connected'}</span></div><p>{cloud?.region??'Cloud GPU resources'}</p><small>{cloud?.status==='ready' ? `${compute.data?.gpu_options?.filter(option=>option.available).map(option=>option.label).join(' · ')||'Cloud GPUs configured'}` : cloud?.message??'Connect Google Cloud to train on a larger GPU.'}</small><button className="text-link" onClick={onSettings}>Manage cloud resources →</button></article>
      <article><div className="dashboard-resource-heading"><Icon name="folder" size={20}/><h3>Hugging Face</h3><span className={`resource-state${hf.data?.configured?' ready':''}`}>{hf.isPending?'Loading':hf.isError?'Unavailable':hf.data?.configured?'Connected':'Public access'}</span></div><p>{hf.data?.username??'Datasets & model weights'}</p><small>{hf.data?.configured?'Token configured for private or gated repositories.':'Public datasets work without a token. Add one for private or gated models.'}</small><button className="text-link" onClick={onSettings}>Manage connection →</button></article></div>
      {check.error && <p className="error-notice" role="alert">{check.error.message}</p>}{check.isSuccess && <p className="dashboard-checked" role="status">Resource checks completed. Availability and setup details are shown above.</p>}
    </section>
    <section className="dashboard-recent"><div className="library-heading"><div><h2>Recent datasets</h2><p>Continue from your saved collection.</p></div><button className="text-link" onClick={onDatasets}>View datasets →</button></div>{datasets.isError?<p role="alert">Could not load dataset history.</p>:datasets.isPending?<p role="status">Loading your collection…</p>:!datasets.data?.length?<div className="dashboard-empty"><Icon name="database" size={28}/><p>Import a dataset to start your collection.</p><button className="secondary-button" onClick={onDatasets}>Import a dataset</button></div>:<div className="dashboard-dataset-list">{datasets.data.slice(0,4).map(entry=><button key={entry.id} onClick={()=>onDataset(entry)}><span className="dashboard-dataset-icon"><Icon name="database" size={19}/></span><span><strong>{entry.name}</strong><small>{projects.find(project=>project.id===entry.project_id)?.name} · {entry.source==='local'?'Local files':'Hugging Face inspection'}</small></span><Icon name="arrow" size={17}/></button>)}</div>}</section>
  </div>;
}
