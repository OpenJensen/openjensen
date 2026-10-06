'use client';

import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { datasetDownload, datasetLibrary, formatName, sampleImage, type DatasetLabels, type LibraryDataset } from '@/lib/dataset-library';
import { api, apiMediaUrl } from '@/lib/api';
import { Icon } from './icon';
import './dataset-library.css';

export function DatasetLibrary({ projectId, onOpen, active }: {active:boolean;projectId:string; onOpen:(entry:LibraryDataset)=>void}) {
  const datasets=useQuery({queryKey:['datasets',projectId],queryFn:()=>datasetLibrary.list(projectId),enabled:!!projectId&&active,retry:false,refetchInterval:active?5000:false});
  const [search,setSearch]=useState('');
  const entries=(datasets.data??[]).filter(entry=>entry.project_id===projectId && entry.name.toLowerCase().includes(search.toLowerCase()));
  return <section className="dataset-library" aria-labelledby="dataset-library-title">
    <div className="library-heading"><h2 id="dataset-library-title">My datasets</h2></div>
    {!!datasets.data?.length && <input className="library-search" aria-label="Search my datasets" placeholder="Search your datasets" value={search} onChange={event=>setSearch(event.target.value)} />}
    {datasets.isError ? <div role="alert"><p>Could not load your datasets.</p><button className="secondary-button" onClick={()=>void datasets.refetch()}>Retry datasets</button></div> : datasets.isPending && projectId ? <p role="status">Loading your datasets…</p> : !entries.length ? <div className="dataset-empty"><Icon name="database" size={32}/><h3>{search ? 'No matching datasets' : 'Your dataset collection starts here'}</h3><p>{search ? 'Try a different name.' : 'Imported and inspected datasets appear here, ready to revisit.'}</p></div> : <div className="dataset-library-grid">{entries.map(entry=><DatasetCard key={entry.id} entry={entry} active={active} onOpen={()=>onOpen(entry)}/>)}</div>}
  </section>;
}

function DatasetCover({ entry, active }: { entry: LibraryDataset; active: boolean }) {
  const local = !entry.id.startsWith('inspection:') && entry.status === 'ready';
  const hub = entry.source === 'huggingface' && !!entry.job_id && entry.status === 'ready';
  const samples = useQuery({ queryKey: ['dataset-samples', entry.id], queryFn: () => datasetLibrary.samples(entry.id), enabled: active && local, retry: false });
  const episodes = useQuery({ queryKey: ['dataset-episodes', entry.job_id, 0, 1], queryFn: () => api.episodes(entry.job_id!, 0, 1), enabled: active && hub, staleTime: Infinity, retry: false });
  const index = episodes.data?.episodes[0]?.episode_index;
  const preview = useQuery({ queryKey: ['dataset-episode', entry.job_id, index], queryFn: () => api.episode(entry.job_id!, index!), enabled: active && hub && index !== undefined, staleTime: Infinity, retry: false });
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const sample = samples.data?.[0];
  const camera = preview.data?.cameras[0];
  const loading = !failed && ((local && samples.isPending) || (hub && episodes.isPending) || (hub && index !== undefined && preview.isPending) || (!!(sample || camera) && !loaded));
  return <div className="dataset-library-cover">
    {active && sample && !failed && <img src={sampleImage(entry.id, sample.path)} alt={`${entry.name} · ${sample.camera}`} onLoad={() => setLoaded(true)} onError={() => setFailed(true)} />}
    {active && camera && !failed && <video src={apiMediaUrl(camera.url)} preload="metadata" muted playsInline aria-label={`${entry.name} preview`} className={loaded ? 'loaded' : ''}
      onLoadedMetadata={event => { event.currentTarget.currentTime = camera.start_seconds; }}
      onLoadedData={event => { if (Math.abs(event.currentTarget.currentTime - camera.start_seconds) < .05) setLoaded(true); }}
      onSeeked={() => setLoaded(true)} onError={() => setFailed(true)} />}
    {!loaded || failed ? <div className="dataset-cover-placeholder"><Icon name="database" size={32}/><small>{loading ? 'Loading preview…' : 'Preview unavailable'}</small></div> : null}
    <span>{entry.example ? 'Synthetic example' : entry.source === 'huggingface' ? 'Hugging Face' : 'Local import'}</span>
  </div>;
}

function DatasetCard({entry,onOpen,active}:{entry:LibraryDataset;onOpen:()=>void;active:boolean}) {
  const stats=entry.converted??entry.detection;
  const episodes=entry.profile?.total_episodes??stats?.episodes;
  const cameras=entry.profile ? Object.values(entry.profile.features).filter(feature=>['video','image'].includes(String((feature as {dtype?:string}).dtype))).length : stats?.cameras?.length;
  return <button type="button" className="dataset-library-card" aria-label={`Open dataset ${entry.name}`} onClick={onOpen}>
    <DatasetCover entry={entry} active={active}/>
    <div className="dataset-library-copy"><h3>{entry.name}</h3><p>{formatName(entry.profile?.format??stats?.format)}</p><div className="dataset-card-facts"><span>{episodes!=null ? `${episodes} episodes` : 'Format detected'}</span>{!!cameras && <span>{cameras} {cameras===1?'view':'views'}</span>}</div><div className="dataset-card-footer"><span>{entry.status==='converting' ? 'Converting…' : entry.status==='failed' ? 'Needs attention' : entry.training_copy ? 'Training copy saved' : entry.profile?.source==='huggingface' ? 'Inspection saved' : entry.status==='ready' ? 'Files saved' : 'Ready to convert'}</span><Icon name="arrow" size={15}/></div></div>
  </button>;
}

export function LocalDatasetImport({projectId,value,onChange,disabled}:{projectId:string;value:LibraryDataset|null;onChange:(entry:LibraryDataset)=>void;disabled:boolean}) {
  const folder=useRef<HTMLInputElement>(null),archive=useRef<HTMLInputElement>(null);
  const [progress,setProgress]=useState('');
  const [fps,setFps]=useState(30),[task,setTask]=useState('Recorded robot task'),[robot,setRobot]=useState('unspecified');
  const [stateKeys,setStateKeys]=useState(''),[actionKey,setActionKey]=useState('');
  const client=useQueryClient();
  const upload=useMutation({mutationFn:(files:File[])=>datasetLibrary.upload(projectId,files,setProgress),onSuccess:entry=>{onChange(entry);void client.invalidateQueries({queryKey:['datasets']});},onSettled:()=>setProgress('')});
  const saved=useQuery({queryKey:['dataset-import',value?.id],queryFn:()=>datasetLibrary.get(value!.id),enabled:!!value && !value.id.startsWith('inspection:'),retry:false,refetchInterval:query=>query.state.data?.status==='converting' ? 1000 : false});
  const current=saved.data??value;
  useEffect(()=>{if(saved.data)onChange(saved.data);},[saved.data,onChange]);
  useEffect(()=>{setFps(value?.detection?.fps??30);setStateKeys(value?.detection?.mapping.state?.join(', ')??'');setActionKey(value?.detection?.mapping.action??'');},[value?.id]);
  const convert=useMutation({mutationFn:()=>datasetLibrary.convert(current!.id,{fps,task,robot_type:robot,mapping:current?.detection?.mapping.action ? {...current.detection.mapping,action:actionKey,state:stateKeys.split(',').map(k=>k.trim()).filter(Boolean)} : undefined}),onSuccess:entry=>{client.setQueryData(['dataset-import',entry.id],entry);onChange(entry);void client.invalidateQueries({queryKey:['datasets']});}});
  function choose(event:ChangeEvent<HTMLInputElement>) { const files=Array.from(event.target.files??[]);event.target.value='';if(files.length)upload.mutate(files); }
  const busy=upload.isPending || current?.status==='converting' || convert.isPending;
  return <div className="local-dataset-import">
    <input ref={folder} className="visually-hidden" type="file" multiple aria-label="Choose dataset folder" {...({webkitdirectory:'',directory:''} as Record<string,string>)} onChange={choose}/>
    <input ref={archive} className="visually-hidden" type="file" accept=".zip,.h5,.hdf5" aria-label="Choose dataset archive or HDF5" onChange={choose}/>
    <div className="local-upload-zone"><Icon name="folder" size={29}/><strong>Choose files from your computer</strong><p>Folder, ZIP or HDF5 · up to 2 GB</p><div className="local-upload-buttons"><button type="button" className="secondary-button" disabled={disabled||busy} onClick={()=>folder.current?.click()}>Choose folder</button><button type="button" className="text-button" disabled={disabled||busy} onClick={()=>archive.current?.click()}>ZIP / HDF5 file</button></div></div>
    {progress && <p className="upload-progress" role="status">{progress}</p>}
    {upload.error && <p className="error-notice" role="alert">{upload.error.message}</p>}
    {current && <section className="detected-dataset" aria-label="Detected dataset"><div><strong>{current.name}</strong><span className="metadata-badge">{formatName(current.detection?.format)}</span></div><p>{current.detection?.file_count} files · {((current.detection?.total_bytes??0)/1024**2).toFixed(1)} MB</p>
      {!current.detection?.convertible && <p role="alert">{current.detection?.reason}</p>}
      {current.status==='converting' && <p role="status">Converting synchronized frames and camera views… You can leave this page; the import is saved.</p>}
      {current.status==='ready' ? <p className="local-import-ready" role="status"><Icon name="check" size={16}/> Ready to inspect · {formatName(current.converted?.format??current.detection?.format)}</p> : current.detection?.convertible && current.status!=='converting' && <div className="conversion-form">
        <label>Frame rate (FPS)<input type="number" min={1} max={240} value={fps} onChange={event=>setFps(Number(event.target.value))}/></label>
        <label>Task description<input value={task} maxLength={1000} onChange={event=>setTask(event.target.value)}/></label>
        <details><summary>Field mapping & robot</summary><label>Robot type<input value={robot} onChange={event=>setRobot(event.target.value)}/></label>{current.detection.mapping.action && <><label>Action field<input value={actionKey} onChange={event=>setActionKey(event.target.value)}/></label><label>State fields (in order, comma separated)<input value={stateKeys} onChange={event=>setStateKeys(event.target.value)}/></label><p>Camera fields: {current.detection.cameras.join(', ')}</p></>}</details>
        <button type="button" className="secondary-button" disabled={disabled||busy||!task.trim()||fps<1||fps>240} onClick={()=>convert.mutate()}>Convert to LeRobot <Icon name="arrow" size={15}/></button>
      </div>}
      {(current.error||convert.error||saved.error) && <p className="error-notice" role="alert">{current.error||convert.error?.message||saved.error?.message}</p>}
    </section>}
  </div>;
}

export function DatasetLabeling({entry,onInspect}:{entry:LibraryDataset;onInspect:()=>void}) {
  const client=useQueryClient();
  const heading=useRef<HTMLHeadingElement>(null);
  useEffect(()=>{heading.current?.focus({preventScroll:true});},[]);
  const saved=useQuery({queryKey:['dataset-import',entry.id],queryFn:()=>datasetLibrary.get(entry.id),retry:false,refetchInterval:query=>query.state.data?.status==='converting'?1000:false});
  const current=saved.data??entry;
  const samples=useQuery({queryKey:['dataset-samples',entry.id],queryFn:()=>datasetLibrary.samples(entry.id),enabled:current.status==='ready',retry:false});
  const [labels,setLabels]=useState<DatasetLabels>(entry.annotations??{views:{},frames:{}});
  const [revision,setRevision]=useState(entry.annotation_revision??0),[dirty,setDirty]=useState(false),[selected,setSelected]=useState(0);
  useEffect(()=>{if(saved.data && !dirty){setLabels(saved.data.annotations??{views:{},frames:{}});setRevision(saved.data.annotation_revision??0);}},[saved.data,dirty]);
  const save=useMutation({mutationFn:()=>datasetLibrary.annotate(entry.id,revision,labels),onSuccess:value=>{setRevision(value.annotation_revision??0);setDirty(false);client.setQueryData(['dataset-import',entry.id],value);void client.invalidateQueries({queryKey:['datasets']});}});
  const sample=samples.data?.[selected];
  const sampleKey=sample ? `${sample.episode_index}:${sample.frame_index}:${sample.camera}` : '';
  const cameras=current.converted?.cameras??current.detection?.cameras??[];
  return <section className="panel dataset-labeling" aria-label="Dataset labeling"><div className="library-heading"><div><h2 ref={heading} tabIndex={-1}>{entry.name}</h2><p>{current.example?'Synthetic example · ':' '}{formatName(current.converted?.format??current.detection?.format)}</p></div>{current.status==='ready' && <a className="secondary-button" href={datasetDownload(entry.id)} download aria-disabled={dirty || save.isPending} onClick={event=>{if(dirty || save.isPending)event.preventDefault();}}>{dirty ? 'Save labels to export' : 'Export dataset'}</a>}</div>
    {current.status==='converting' && <p role="status">Preparing camera views…</p>}
    {current.error && <p className="error-notice" role="alert">{current.error}</p>}
    {current.status==='ready' && <><div className="dataset-label-grid"><div className="label-image-area">{sample ? <><img src={sampleImage(entry.id,sample.path)} alt={`${labels.views[sample.camera]||sample.camera} · episode ${sample.episode_index+1} frame ${sample.frame_index}`} /><div className="sample-navigation"><button className="secondary-button" disabled={selected===0} onClick={()=>setSelected(i=>i-1)}>Previous image</button><span>{selected+1} / {samples.data?.length}</span><button className="secondary-button" disabled={selected===(samples.data?.length??0)-1} onClick={()=>setSelected(i=>i+1)}>Next image</button></div><p>Episode {sample.episode_index+1} · Frame {sample.frame_index} · {labels.views[sample.camera]||sample.camera}</p></> : <p>{samples.isPending?'Loading sample frames…':current.preview_note||'No preview image available for this dataset.'}</p>}{samples.error && <p role="alert">{samples.error.message}</p>}</div>
      <div className="dataset-label-fields"><h3>Camera views</h3>{cameras.map(camera=><label key={camera}>{camera.split(/[/.]/).at(-1)?.replaceAll('_',' ')} view<input aria-label={`View name ${camera}`} value={labels.views[camera]??''} maxLength={100} placeholder="e.g. Front camera, left wrist" disabled={save.isPending} onChange={event=>{setDirty(true);setLabels({...labels,views:{...labels.views,[camera]:event.target.value}});}}/></label>)}
      {sample && <label>Label this image<textarea aria-label="Image label" value={labels.frames[sampleKey]??''} maxLength={1000} placeholder="Object, action, task or a note about this frame" disabled={save.isPending} onChange={event=>{setDirty(true);setLabels({...labels,frames:{...labels.frames,[sampleKey]:event.target.value}});}}/></label>}
      <button className="primary-button" disabled={!dirty||save.isPending} onClick={()=>save.mutate()}>{save.isPending?'Saving…':'Save labels'}</button>{save.isSuccess&&!dirty && <p role="status">Labels saved</p>}{save.error && <div role="alert"><p className="error-notice">{save.error.message}</p><button className="text-button" onClick={()=>{setDirty(false);void saved.refetch();}}>Reload saved labels</button></div>}
      <p className="field-help">Save labels before exporting. The export includes LeRobot files and your camera/image annotations.</p></div></div><div className="dataset-label-actions"><button className="secondary-button" onClick={onInspect}>Inspect for training <Icon name="arrow" size={15}/></button></div></>}
  </section>;
}
