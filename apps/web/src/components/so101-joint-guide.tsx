'use client';

import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { robotJoints, type RobotJoint } from '@/lib/robot-api';
import { add3, jointPose, normalize3, previewRanges, projectJoint, scale3, transformVector } from '@/lib/so101-schematic';
import { So101CadView } from './so101-cad-view';
import './so101-joint-guide.css';

const axes:Record<RobotJoint,string>={shoulder_pan:'Vertical rotation',shoulder_lift:'Shoulder pitch',elbow_flex:'Elbow pitch',wrist_flex:'Wrist pitch',wrist_roll:'Tool rotation',gripper:'Jaw hinge'};
const markerOffsets:Record<RobotJoint,[number,number]>={shoulder_pan:[-44,25],shoulder_lift:[-42,-2],elbow_flex:[-34,-25],wrist_flex:[0,40],wrist_roll:[10,-40],gripper:[45,-20]};
const reference=jointPose();
const names=robotJoints.map(joint=>joint.name);

export function So101JointGuide({selected,onSelect,calibrationActive=false,detecting=false,setupActive=false,measured,feedback}:{selected:RobotJoint;onSelect:(joint:RobotJoint)=>void;calibrationActive?:boolean;detecting?:boolean;setupActive?:boolean;measured?:{value:number;min?:number;max?:number};feedback?:ReactNode}) {
 const id=useId().replaceAll(':',''),buttons=useRef<Partial<Record<RobotJoint,HTMLButtonElement|null>>>({});
 const [angle,setAngle]=useState(0),[playing,setPlaying]=useState(false),[reduced,setReduced]=useState(true),[preview,setPreview]=useState(0);
 const joint=robotJoints.find(joint=>joint.name===selected)!,pose=useMemo(()=>jointPose(selected,angle),[selected,angle]),frame=pose.joints[selected],pivot=projectJoint(frame.origin),range=previewRanges[selected];
 const axisA=projectJoint(add3(frame.origin,scale3(frame.axis,-.044))),axisB=projectJoint(add3(frame.origin,scale3(frame.axis,.044)));
 const x=normalize3(transformVector(frame.matrix,[1,0,0])),y=normalize3(transformVector(frame.matrix,[0,1,0])),radius=selected==='gripper'?.032:.044;
 const arc=Array.from({length:24},(_,i)=>{const theta=(-65+i*130/23)*Math.PI/180;const p=projectJoint(add3(frame.origin,add3(scale3(x,Math.cos(theta)*radius),scale3(y,Math.sin(theta)*radius))));return `${p.x},${p.y}`;}).join(' ');
 useEffect(()=>{const media=window.matchMedia('(prefers-reduced-motion: reduce)'),update=()=>{setReduced(media.matches);if(media.matches)setPlaying(false);};update();media.addEventListener('change',update);return()=>media.removeEventListener('change',update);},[]);
 useEffect(()=>{setPlaying(false);setAngle(0);},[selected]);
 useEffect(()=>{
  if(!playing||reduced)return;
  let frameId=0,started=0;
  const animate=(now:number)=>{if(!started)started=now;const elapsed=now-started;if(elapsed>=1800){setAngle(0);setPlaying(false);return;}const t=elapsed/1800,weight=Math.sin(t*Math.PI*2);setAngle(weight>=0?weight*range[1]:-weight*range[0]);frameId=requestAnimationFrame(animate);};
  frameId=requestAnimationFrame(animate);return()=>cancelAnimationFrame(frameId);
 },[playing,reduced,preview,selected,range]);
 function keyboard(event:KeyboardEvent<HTMLButtonElement>,name:RobotJoint){const i=names.indexOf(name),next=event.key==='ArrowRight'||event.key==='ArrowDown'?(i+1)%6:event.key==='ArrowLeft'||event.key==='ArrowUp'?(i+5)%6:event.key==='Home'?0:event.key==='End'?5:-1;if(next<0)return;event.preventDefault();onSelect(names[next]);buttons.current[names[next]]?.focus();}
 return <section className="so101-guide" aria-label="SO-101 joint illustration" data-selected-joint={selected} data-motion={reduced?'reduced':playing?'playing':angle===0?'static':'paused'} data-angle={angle.toFixed(2)}>
  <header className="so101-guide-header"><h3><span aria-hidden="true">{joint.id}</span>{joint.label}</h3><span>{axes[selected]}</span><small>CAD · not live</small></header>
  <div className="so101-cad-scene"><So101CadView pose={pose} selected={selected}/><svg className="so101-cad-overlay" viewBox="0 0 600 420" aria-hidden="true"><defs><marker id={`${id}-arrow`} markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto-start-reverse"><path d="M1 1L6 3.5L1 6"/></marker></defs><line className="so101-cad-axis" x1={axisA.x} y1={axisA.y} x2={axisB.x} y2={axisB.y}/><polyline className="so101-cad-rotation" points={arc} markerEnd={`url(#${id}-arrow)`}/><circle className="so101-cad-pivot" cx={pivot.x} cy={pivot.y} r="3"/>{robotJoints.map(item=>{const anchor=projectJoint(reference.joints[item.name].origin),position=projectJoint(pose.joints[item.name].origin),offset=markerOffsets[item.name];return <line className={`so101-cad-leader${item.name===selected?' selected':''}`} key={item.name} x1={anchor.x+offset[0]} y1={anchor.y+offset[1]} x2={position.x} y2={position.y}/>;})}</svg>
   <div className="so101-guide-markers" role="group" aria-label="Choose an illustrated joint">{robotJoints.map(item=>{const p=projectJoint(reference.joints[item.name].origin),offset=markerOffsets[item.name];return <button key={item.name} ref={element=>{buttons.current[item.name]=element;}} type="button" style={{left:`${(p.x+offset[0])/600*100}%`,top:`${(p.y+offset[1])/420*100}%`}} className={`so101-guide-marker${selected===item.name?' selected':''}`} aria-label={`Joint ${item.id}: ${item.label}`} aria-pressed={selected===item.name} onClick={()=>onSelect(item.name)} onKeyDown={event=>keyboard(event,item.name)}>{item.id}</button>;})}</div>
  </div>
  <div className="so101-guide-controls"><button type="button" className="secondary-button" disabled={reduced} title={reduced?'Motion preview is disabled by your reduced-motion setting. Use the angle slider instead.':undefined} aria-label={playing?'Pause motion':'Preview motion'} onClick={()=>{if(playing)setPlaying(false);else{setAngle(0);setPreview(value=>value+1);setPlaying(true);}}}>{playing?'Pause motion':'Preview motion'}</button><button type="button" className="secondary-button" disabled={angle===0&&!playing} onClick={()=>{setPlaying(false);setAngle(0);}}>Reset</button><label><span>Angle</span><input type="range" aria-label="Illustrated joint angle" min={range[0]} max={range[1]} step="1" value={angle} onChange={event=>{setPlaying(false);setAngle(Number(event.target.value));}}/><output>{angle>0?'+':''}{angle.toFixed(0)}°</output></label></div>
  {(calibrationActive||detecting||setupActive)&&<p className="so101-guide-context">{calibrationActive?'Power on. Support the arm and move the selected joint gently.':detecting?'Power on. Move only this joint gently, hold its new position, then confirm.':setupActive?'Keep power on and support the arm. Wait for identification.':'Keep power on and support the arm. Wait for identification.'}</p>}
  {feedback}
  {measured&&<div className="so101-guide-measured" role="group" aria-label="Measured encoder values"><span>Encoder</span><output>{Number.isFinite(measured.value)?measured.value.toFixed(0):'—'} ticks</output>{measured.min!==undefined&&measured.max!==undefined&&<small>Range {measured.min.toFixed(0)}–{measured.max.toFixed(0)}</small>}</div>}
 </section>;
}
