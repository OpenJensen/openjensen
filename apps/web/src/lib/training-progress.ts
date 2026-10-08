import type { JobEvent } from './api';

const stages = [
  { id: 'accepted', label: 'Request received' },
  { id: 'compute', label: 'Allocating compute' },
  { id: 'environment', label: 'Preparing environment' },
  { id: 'inputs', label: 'Loading data and model' },
  { id: 'training', label: 'Training' },
  { id: 'verification', label: 'Saving and verifying' },
  { id: 'ready', label: 'Ready' },
] as const;

export function trainingProgress(status: string, phase: string, currentAction: string | undefined, events: JobEvent[]) {
  function stageOf(stage: string, message = '') {
    if (stage === 'verifying' || /reload|verif|final.*upload|upload.*checkpoint/i.test(message)) return 5;
    if (['training', 'validation', 'checkpoint'].includes(stage)) return 4;
    if (/download|loading.*(?:model|dataset|backbone|tokenizer)|training statistics|snapshot|upload.*(?:input|model|dataset)/i.test(message)) return 3;
    if (/install|dependenc|environment|setup|compil|worker connected/i.test(message)) return 2;
    if (/provision|allocat|capacity|launch|GPU|cloud storage/i.test(message)) return 1;
    return stage === 'preparing' || stage === 'operation' ? 1 : 0;
  }
  const observations = events.map(event => ({ index: stageOf(event.stage, event.message), message: event.message, time: event.timestamp }));
  // Checkpoint/validation events repeat during training; they never imply readiness.
  const observed = Math.max(0, ...observations.map(event => event.index));
  const complete = status === 'succeeded';
  const stopped = ['failed', 'cancelled', 'interrupted'].includes(status);
  const active = complete ? 6 : Math.max(observed, stageOf(phase, currentAction));
  return stages.map((stage, index) => {
    const event = observations.filter(item => item.index === index).at(-1);
    return { ...stage, state: complete || index < active ? 'complete' : index > active ? 'pending' : stopped ? 'stopped' : 'active',
      message: index === active && currentAction ? currentAction : event?.message,
      timestamp: event?.time };
  });
}
