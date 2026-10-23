'use client';
import { useEffect, useRef, useState } from 'react';
import type { RobotJoint } from '@/lib/robot-api';
import { createSo101Renderer, loadSo101Meshes, type CadRenderer } from '@/lib/so101-webgl';
import { jointOrder, projectJoint, type Camera, type RobotPose } from '@/lib/so101-schematic';

export function So101CadView({pose,selected,view}:{pose:RobotPose;selected:RobotJoint;view:Camera}) {
 const canvas=useRef<HTMLCanvasElement>(null),renderer=useRef<CadRenderer|null>(null),latest=useRef({pose,selected,view});latest.current={pose,selected,view};
 const [state,setState]=useState<'loading'|'webgl'|'fallback'>('loading');
 useEffect(()=>{
  let active=true,loaded:Awaited<ReturnType<typeof loadSo101Meshes>>|null=null;
  const element=canvas.current;
  const draw=()=>{try{renderer.current?.render(latest.current.pose,latest.current.selected,latest.current.view);}catch{renderer.current?.dispose();renderer.current=null;if(active)setState('fallback');}};
  const initialize=()=>{
   if(!active||!element||!loaded)return;
   try{renderer.current?.dispose();renderer.current=createSo101Renderer(element,loaded);renderer.current.render(latest.current.pose,latest.current.selected,latest.current.view);setState('webgl');}catch{renderer.current?.dispose();renderer.current=null;setState('fallback');}
  };
  const lost=(event:Event)=>{event.preventDefault();renderer.current?.dispose();renderer.current=null;if(active)setState('fallback');};
  const restored=()=>initialize();
  element?.addEventListener('webglcontextlost',lost);element?.addEventListener('webglcontextrestored',restored);
  void loadSo101Meshes().then(meshes=>{loaded=meshes;initialize();}).catch(()=>{if(active)setState('fallback');});
  const observer=new ResizeObserver(draw);if(element)observer.observe(element);
  const theme=new MutationObserver(draw);theme.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  return()=>{active=false;observer.disconnect();theme.disconnect();element?.removeEventListener('webglcontextlost',lost);element?.removeEventListener('webglcontextrestored',restored);renderer.current?.dispose();renderer.current=null;};
 },[]);
 useEffect(()=>{try{renderer.current?.render(pose,selected,view);}catch{renderer.current?.dispose();renderer.current=null;setState('fallback');}},[pose,selected,view]);
 const points=jointOrder.map(name=>projectJoint(pose.joints[name].origin,view));
 return <><canvas ref={canvas} className="so101-cad-canvas" aria-hidden="true" data-renderer={state}/>{state!=='webgl'&&<svg className="so101-cad-fallback" viewBox="0 0 600 420" aria-hidden="true"><polyline points={points.map(p=>`${p.x},${p.y}`).join(' ')} fill="none" stroke="currentColor" strokeWidth="8" strokeLinejoin="miter"/>{points.map((p,i)=><rect key={i} x={p.x-8} y={p.y-11} width="16" height="22" rx="2"/>)}<text x="20" y="400">{state==='loading'?'Loading CAD…':'Joint frames · CAD view unavailable'}</text></svg>}</>;
}
