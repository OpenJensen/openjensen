/** Replaceable Pi runtime: NDJSON RPC over private stdin/stdout, no built-in tools. */
import readline from "node:readline";
import { Agent } from "@earendil-works/pi-agent-core";
import { createModels, createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { openaiProvider } from "@earendil-works/pi-ai/providers/openai";
import { Type } from "@sinclair/typebox";

const send = (value) => process.stdout.write(JSON.stringify(value) + "\n");
const input = readline.createInterface({ input: process.stdin });
const pending = new Map();
let agent, started = false;
input.on("line", (line) => {
  const value = JSON.parse(line);
  if (value.type === "start" && !started) {
    started = true;
    run(value).catch(() => { send({type:"fatal", error:"Pi runtime failed; inspect saved tool receipts"}); process.exitCode=1; input.close(); });
  } else if (value.type === "result") {
    pending.get(value.id)?.(value);
    pending.delete(value.id);
  } else if (value.type === "cancel") agent?.abort();
});
input.on("close", () => agent?.abort());
process.on("SIGTERM", () => { agent?.abort(); setTimeout(() => process.exit(1), 500).unref(); });

async function run(config) {
  const rpc = async (name, id, args, signal) => {
    if (signal?.aborted) throw new Error("Cancelled");
    return await new Promise((resolve) => {
      const aborted = () => {
        pending.delete(id);
        resolve({content:[{type:"text",text:"Cancelled"}],details:{cancelled:true},isError:true,terminate:true});
      };
      signal?.addEventListener("abort", aborted, {once:true});
      pending.set(id, (value) => {
        signal?.removeEventListener("abort", aborted);
        resolve({
          content: value.content ?? [{type:"text", text:JSON.stringify(value.result)}],
          details: value.result,
          isError: !!value.error,
          terminate: !!value.error || name === "stop",
        });
      });
      send({type:"tool", name, id, args});
    });
  };
  const empty = Type.Object({}, {additionalProperties:false});
  const names = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"];
  const tools = [
    {name:"observe", description:"Get paired wrist and side images and measured joint state. Observe before every move.", parameters:empty},
    {name:"move", description:"Propose one small joint-position segment. Supervisor approval required. Degrees; gripper percent. No Cartesian or collision planning.", parameters:Type.Object({
      observation_id:Type.String(), targets:Type.Object(Object.fromEntries(names.map(n=>[n,Type.Optional(Type.Number())])), {additionalProperties:false}),
      duration_seconds:Type.Number({minimum:0.5,maximum:5})
    }, {additionalProperties:false})},
    {name:"set_gripper", description:"Propose a gripper segment, at most 10 percent from measured state.", parameters:Type.Object({
      observation_id:Type.String(), target:Type.Number({minimum:0,maximum:100}), duration_seconds:Type.Number({minimum:0.5,maximum:5})
    }, {additionalProperties:false})},
    {name:"stop", description:"Stop and save this episode. Task success is unverified until human review.", parameters:empty},
  ].map(t=>({...t,label:t.name,executionMode:"sequential",execute:(id,args,signal)=>rpc(t.name,id,args,signal)}));
  const models = createModels();
  models.setProvider(openaiProvider());
  // Explicit model descriptor keeps the adapter independent of Pi's catalog age.
  const model = models.getModel("openai", config.model) ?? {
    id:config.model, name:config.model, api:"openai-responses", provider:"openai",
    baseUrl:"https://api.openai.com/v1", reasoning:true, input:["text","image"],
    contextWindow:1050000, maxTokens:128000,
    cost:{input:10,output:50,cacheRead:1,cacheWrite:12.5},
  };
  let turn=0, tokens=0;
  const mockStream = (_model, context) => {
    const stream = createAssistantMessageEventStream();
    const observations = context.messages.filter(m=>m.role==="toolResult" && m.toolName==="observe");
    const observed = observations.length ? JSON.parse(observations.at(-1).content.find(c=>c.type==="text").text) : null;
    const actions = [
      ["observe",{}],
      ["move",{observation_id:observed?.observation_id,targets:{shoulder_pan:(observed?.joints.shoulder_pan??0)+2},duration_seconds:1.2}],
      ["observe",{}],
      ["set_gripper",{observation_id:observed?.observation_id,target:(observed?.joints.gripper??50)-3,duration_seconds:1.2}],
      ["observe",{}],
      ["stop",{}],
    ];
    const [name,args] = actions[turn] ?? ["stop",{}];
    const message={role:"assistant",content:[{type:"toolCall",id:"mock_"+turn,name,arguments:args}],api:"openai-responses",provider:"mock",model:"deterministic-pi",usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}},stopReason:"toolUse",timestamp:Date.now()};
    queueMicrotask(()=>{stream.push({type:"start",partial:message});stream.push({type:"done",reason:"toolUse",message});});
    return stream;
  };
  const prompt = "Collect one supervised tabletop pick-place demonstration. Only use registered tools. Observe both views before proposing a segment. Never infer Cartesian coordinates from joint angles. Use small bounded moves, stop on uncertainty, contact, missing views or failed tools. All physical moves require operator approval. Never claim task completion from command acceptance. Save contact, settling and thinking intervals. Task: " + config.task;
  agent = new Agent({
    initialState:{model,systemPrompt:prompt,tools,thinkingLevel:"low"},
    streamFn:config.mode==="mock" ? mockStream : (m,c,o)=>models.streamSimple(m,c,{...o,apiKey:config.api_key,maxTokens:Math.min(2048,config.max_tokens-tokens),maxRetries:0,reasoning:"low"}),
    toolExecution:"sequential",
    beforeToolCall:async()=> turn >= config.max_steps || tokens >= config.max_tokens ? {block:true,reason:"Run budget reached",terminate:true} : undefined,
    prepareRequest:async()=>{if(turn>=config.max_steps || tokens>=config.max_tokens) throw new Error("Run budget reached");},
    finishTurn:async({message})=> {
      turn++;
      tokens += message.usage?.totalTokens ?? 0;
      return turn>=config.max_steps || tokens>=config.max_tokens ? {action:"end"} : undefined;
    },
  });
  send({type:"configuration",runtime:"pi-agent-core",version:"1.5.1",model:config.mode==="mock"?"deterministic-pi":config.model,provider:config.mode==="mock"?"mock":"openai",prompt,tools:tools.map(t=>({name:t.name,parameters:t.parameters}))});
  agent.subscribe((event)=>{
    // Persist exposed text and tool requests/results only. No thinking blocks,
    // encrypted reasoning, provider auth, or image base64 in lifecycle logs.
    if(event.type==="message_end" && event.message.role==="assistant"){
      const m=event.message;
      send({type:"model_output",model:m.model,provider:m.provider,content:m.content.filter(c=>c.type==="text"||c.type==="toolCall").map(c=>c.type==="text"?{type:"text",text:c.text}:{type:"toolCall",id:c.id,name:c.name,arguments:c.arguments}),usage:m.usage,stopReason:m.stopReason});
    }
    if(["agent_start","turn_start","turn_end","agent_end"].includes(event.type)) send({type:"lifecycle",event:event.type,turn});
  });
  const timeout=setTimeout(()=>agent.abort(),config.max_seconds*1000);
  try { await agent.prompt(config.task); send({type:"done",error:agent.state.errorMessage??null,turns:turn,tokens}); }
  finally {clearTimeout(timeout); input.close();}
}
