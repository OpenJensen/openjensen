export type TrainingModel = {
  id: string;
  label: string;
  description: string;
  model_id: string;
  model_revision: string | null;
  methods: string[];
  checkpoint_subdirectory?: string | null;
  available?: boolean;
  status?: 'ready' | 'setup_required' | 'coming_soon' | 'connect_account';
  unavailable_reason?: string | null;
  runtime_ids?: string[];
  backend?: string;
  required_cameras?: number | null;
  minimum_gpu_memory_gb?: number | null;
  suggested_gpu_memory_gb?: number | null;
  gradient_accumulation_supported?: boolean;
  gradient_accumulation_runtime_ids?: string[];
  training_step_unit?: string;
  training_world_size?: number;
};

// Display families follow the upstream LeRobot policy guides. VLA-JEPA is
// still an action policy: its separate group describes its world-model component.
// https://github.com/huggingface/lerobot/blob/main/README.md
// https://huggingface.co/docs/lerobot/vla_jepa
const modelFamilies = [
  { id: 'vla', label: 'Vision-language-action models', models: ['smolvla', 'openvla', 'openvla_oft', 'pi0', 'pi05', 'pi0_fast', 'gr00t_n17', 'eo1', 'evo1', 'wall_x', 'xvla', 'psi0'] },
  { id: 'action-sequence', label: 'Action sequence policies', models: ['act', 'vqbet'] },
  { id: 'diffusion', label: 'Diffusion policies', models: ['diffusion', 'multi_task_dit'] },
  { id: 'world-model', label: 'World-model-assisted VLA', models: ['vla_jepa'] },
];

export function groupedTrainingModels(models: TrainingModel[]) {
  const available = models.filter(model => model.status !== 'coming_soon');
  const known = new Set(modelFamilies.flatMap(family => family.models));
  return [
    ...modelFamilies.map(family => ({ id: family.id, label: family.label, models: available.filter(model => family.models.includes(model.id)) })),
    { id: 'other', label: 'Other policies', models: available.filter(model => !known.has(model.id)) },
    { id: 'coming-soon', label: 'Coming soon', models: models.filter(model => model.status === 'coming_soon') },
  ].filter(group => group.models.length);
}

/** Missing or partial catalog support must never opt an older backend into accumulation. */
export function gradientAccumulationAvailable(model: TrainingModel | undefined, runtimeId: string | undefined, method: string | undefined): boolean {
  if (!model || !runtimeId || !method || !model.methods.includes(method)) return false;
  const supportedAdapter = model.id === 'act' ? method === 'full' : model.id === 'smolvla' && ['lora', 'qlora'].includes(method);
  return supportedAdapter && model.gradient_accumulation_supported === true
    && Array.isArray(model.gradient_accumulation_runtime_ids) && model.gradient_accumulation_runtime_ids.includes(runtimeId)
    && model.training_step_unit === 'optimizer_updates' && model.training_world_size === 1;
}

// Keep the picker visible when connected to an older API. Runtime availability
// comes from /policy-options; only its original SmolVLA adapter is assumed here.
export const trainingModels: TrainingModel[] = [
  {
    id: 'smolvla', label: 'SmolVLA', description: 'Compact vision-language-action policy',
    model_id: 'lerobot/smolvla_base', model_revision: 'd9f33c94a60fb382c90dea2164c96845bd955e28',
    methods: ['lora', 'qlora'],
  },
  {
    id: 'openvla_oft', label: 'OpenVLA-OFT', description: 'Continuous action chunks',
    model_id: 'moojink/openvla-7b-oft-finetuned-libero-spatial',
    model_revision: '6d0231af0e48c5985f1ff86908f4674b84bc049b', methods: ['lora', 'qlora'],
    available: false, status: 'coming_soon', unavailable_reason: 'Coming soon', runtime_ids: [],
  },
  {
    id: 'openvla', label: 'OpenVLA', description: 'Autoregressive action prediction',
    model_id: 'openvla/openvla-7b-finetuned-libero-spatial',
    model_revision: '962318cec55ac10993ff0f5f43eda9a270b4c873', methods: ['lora', 'qlora'],
    available: false, status: 'coming_soon', unavailable_reason: 'Coming soon', runtime_ids: [],
  },
  {
    id: 'pi0', label: 'π₀', description: 'Flow-matching robot policy',
    model_id: 'lerobot/pi0_libero_finetuned_v044',
    model_revision: '45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f', methods: ['lora', 'qlora'],
    available: false, status: 'coming_soon', unavailable_reason: 'Coming soon', runtime_ids: [],
  },
  {
    id: 'pi05', label: 'π₀.₅', description: 'OpenPI robot policy',
    model_id: 'gs://openpi-assets/checkpoints/pi05_libero', model_revision: null, methods: ['lora', 'qlora'],
    available: false, status: 'coming_soon', unavailable_reason: 'Coming soon', runtime_ids: [],
  },
  {
    id: 'gr00t_n17', label: 'GR00T N1.7', description: 'NVIDIA robot foundation model',
    model_id: 'nvidia/GR00T-N1.7-LIBERO', model_revision: '2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21',
    checkpoint_subdirectory: 'libero_spatial', methods: ['lora', 'qlora'],
    available: false, status: 'coming_soon', unavailable_reason: 'Coming soon', runtime_ids: [],
  },
];
