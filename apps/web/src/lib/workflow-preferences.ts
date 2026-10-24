export type Preferences = {
  mode: 'engine' | 'libero';
  suite: 'libero_object' | 'libero_spatial';
  taskIds: string;
  parityProfile: string;
  parityRmse: string;
  parityMaxError: string;
  repetitions: number;
  steps: number;
  warmup: number;
  task: number;
  searchStates: string;
  finalStates: string;
  trainingSteps: number;
  camera: string;
  select: boolean;
  minSuccess: number;
  maxLatency: number;
  maxMemory: number;
  precision: 'recommended' | 'Q4_0' | 'Q8_0';
  vision: boolean;
  compareQ4: boolean;
};
export const initialPreferences: Preferences = {
  mode: 'engine',
  suite: 'libero_object',
  taskIds: '0,1,2,3,4,5,6,7,8,9',
  parityProfile: '',
  parityRmse: '',
  parityMaxError: '',
  repetitions: 10,
  steps: 500,
  warmup: 3,
  task: 0,
  searchStates: '0,1',
  finalStates: '2,3',
  trainingSteps: 20000,
  camera: 'observation.images.front',
  select: false,
  minSuccess: 100,
  maxLatency: 1000,
  maxMemory: 8192,
  precision: 'Q8_0',
  vision: false,
  compareQ4: false,
};
