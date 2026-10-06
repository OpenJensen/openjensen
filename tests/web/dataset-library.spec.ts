import { expect, test, type Page } from '@playwright/test';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { mkdir } from 'node:fs/promises';

test.describe.configure({timeout:60000});
async function ownProject(page:Page,name:string) {
  const response=await page.request.post('/api/v1/projects',{data:{name}});
  expect(response.status()).toBe(201);
  const project=await response.json();
  await page.addInitScript(id=>localStorage.setItem('firebird.project',id),project.id);
  await page.goto('/');
  await expect(page.getByLabel('Current project')).toHaveAttribute('data-project-id',project.id);
  return project;
}

test('create a real two-view example, save labels, reopen it and reach the dashboard',async({page},testInfo)=>{
  await ownProject(page,`Label playground ${testInfo.project.name}`);
  await expect(page.getByRole('heading',{name:'My datasets',exact:true})).toBeVisible();
  await expect(page.getByText('Example datasets',{exact:true})).toHaveCount(0);
  await expect(page.getByRole('status',{name:'Application API connection'})).toHaveCount(0);
  await expect(page.getByText('LeRobot v2 / v3',{exact:true})).toHaveCount(0);
  await page.getByRole('button',{name:'Create example',exact:true}).click();
  const labels=page.getByRole('region',{name:'Dataset labeling'});
  await expect(labels.getByRole('heading',{name:'Robot labeling playground'})).toBeVisible();
  const image=labels.getByRole('img');
  await expect(image).toBeVisible();
  await expect.poll(()=>image.evaluate(node=>(node as HTMLImageElement).complete&&(node as HTMLImageElement).naturalWidth>0)).toBe(true);
  const views=labels.locator('input[aria-label^="View name"]');
  await expect(views).toHaveCount(2);
  await views.first().fill('Front camera');
  await labels.getByRole('textbox',{name:'Image label',exact:true}).fill('Red block approaching the target');
  await expect(labels.getByRole('link',{name:'Save labels to export'})).toHaveAttribute('aria-disabled','true');
  await labels.getByRole('button',{name:'Save labels',exact:true}).click();
  await expect(labels.getByText('Labels saved',{exact:true})).toBeVisible();
  await labels.getByRole('button',{name:'Next image',exact:true}).click();
  await expect(labels.getByRole('textbox',{name:'Image label'})).toHaveValue('');
  await labels.getByRole('button',{name:'Previous image',exact:true}).click();
  await expect(labels.getByRole('textbox',{name:'Image label'})).toHaveValue('Red block approaching the target');
  await page.getByRole('button',{name:'My datasets',exact:true}).click();
  await page.getByRole('button',{name:'Open dataset Robot labeling playground'}).click();
  await expect(page.getByRole('textbox',{name:'Image label'})).toHaveValue('Red block approaching the target');
  await expect(labels.getByRole('link',{name:'Export dataset'})).toHaveAttribute('href',/\/datasets\/[a-f0-9]{32}\/download$/);
  await page.screenshot({path:testInfo.outputPath('dataset-labeling.png'),fullPage:true});
  await labels.getByRole('button',{name:'Inspect for training'}).click();
  await expect(page.getByRole('checkbox',{name:'Prepare immutable training copy'})).toBeChecked();
  await page.getByRole('button',{name:'Inspect dataset',exact:true}).click();
  await expect(page.getByText(/Training copy verified/)).toBeVisible();
  await expect(page.getByRole('button',{name:'Train on this dataset'})).toBeEnabled();
  await page.getByRole('button',{name:'Dashboard',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Resources & connections'})).toBeVisible();
  await expect(page.getByRole('button',{name:'My models',exact:true})).toBeVisible();
  await expect(page.getByRole('button',{name:'My datasets',exact:true})).toBeVisible();
  await page.screenshot({path:testInfo.outputPath('workspace-dashboard.png'),fullPage:true});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth+1)).toBe(true);
});

test('native folder selection uploads files, detects records and converts for inspection',async({page},testInfo)=>{
  await ownProject(page,`Folder import ${testInfo.project.name}`);
  const source=resolve(testInfo.outputPath('source-folder'));await mkdir(source,{recursive:true});
  const worker=resolve('packages/core/src/vla_platform/datasets/import_worker.py');
  const reader=resolve('workers/_cpu_readers/.venv/bin/python');
  const result=spawnSync(reader,['-I','-c','import runpy,sys;from pathlib import Path;runpy.run_path(sys.argv[1],run_name="adapter")["demo"](Path(sys.argv[2]))',worker,source],{encoding:'utf8'});
  expect(result.status,result.stderr).toBe(0);
  await page.getByRole('radio',{name:'Local files',exact:true}).check();
  await page.getByLabel('Choose dataset folder',{exact:true}).setInputFiles(source);
  const detected=page.getByRole('region',{name:'Detected dataset'});
  await expect(detected.getByText('Images + records',{exact:true})).toBeVisible();
  await page.getByLabel('Frame rate (FPS)',{exact:true}).fill('6');
  await page.getByLabel('Task description',{exact:true}).fill('Move the block');
  await page.getByRole('button',{name:'Convert to LeRobot'}).click();
  await expect(detected.getByText(/Ready to inspect/)).toBeVisible();
  await page.getByRole('button',{name:'Inspect dataset',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Dataset inspection',exact:true})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Local dataset',exact:true})).toBeVisible();
  await page.getByRole('button',{name:'My datasets',exact:true}).click();
  await expect(page.getByRole('button',{name:'Open dataset source-folder'})).toBeVisible();
});

test('saved Hugging Face inspections reopen without a new inspection request',async({page})=>{
  const time='2026-10-06T08:00:00Z',project='history-project';const writes:string[]=[];
  const profile={source:'huggingface',repo_id:'our/robot-data',revision:'a'.repeat(40),format:'lerobot_v3',robot_type:'test',total_episodes:4,total_frames:40,fps:10,features:{action:{dtype:'float32',shape:[2]}},license:null,metadata_sha256:'b'.repeat(64),inspected_at:time,warnings:[],inspection_scope:'metadata_only'};
  const job={id:'saved-inspection',project_id:project,kind:'dataset.inspect',status:'succeeded',created_at:time,updated_at:time,request:{source:'huggingface',repo_id:'our/robot-data',revision:'a'.repeat(40)},result:profile};
  await page.route('**/api/v1/**',route=>{
    const request=route.request(),path=new URL(request.url()).pathname;
    if(request.method()!=='GET'){writes.push(path);return route.fulfill({status:405,json:{detail:'No writes during reuse'}});}
    if(path==='/api/v1/projects')return route.fulfill({json:[{id:project,name:'History',created_at:time}]});
    if(path.endsWith('/jobs'))return route.fulfill({json:[job]});
    if(path==='/api/v1/datasets')return route.fulfill({json:[{id:'inspection:saved-inspection',job_id:job.id,project_id:project,name:profile.repo_id,source:'huggingface',status:'ready',created_at:time,profile}]});
    if(path.endsWith('/episodes'))return route.fulfill({status:422,json:{detail:'Preview is offline in this test'}});
    return route.continue();
  });
  await page.goto('/');
  await page.getByRole('button',{name:'Open dataset our/robot-data'}).click();
  await expect(page.getByRole('heading',{name:'our/robot-data',exact:true})).toBeVisible();
  expect(writes).toEqual([]);
});
