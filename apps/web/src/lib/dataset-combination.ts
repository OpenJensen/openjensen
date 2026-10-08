import type { DatasetProfile } from './api';

type Feature = { dtype?: unknown; shape?: unknown; names?: unknown };
const feature = (profile: DatasetProfile, key: string): Feature => (profile.features[key] as Feature) ?? {};
function stable(value: unknown): string {
  if (value == null) return "null";
  if (Array.isArray(value)) return `[${value.map(stable).join(',')}]`;
  if (value && typeof value === 'object') return JSON.stringify(Object.keys(value).sort().map(key => [key, stable((value as Record<string, unknown>)[key])]));
  return JSON.stringify(value) ?? 'undefined';
}
const same = (a: unknown, b: unknown) => stable(a) === stable(b);
const imageKeys = (profile: DatasetProfile) => Object.keys(profile.features).filter(key => ['image', 'video'].includes(String(feature(profile, key).dtype)));

/** Prefer matching names, then pair remaining cameras by dimensions without reusing a view. */
export function combinationCameraMapping(reference: DatasetProfile, candidate: DatasetProfile, cameraKeys: string[]): Record<string, string> {
  const remaining = new Set(imageKeys(candidate));
  const mapping: Record<string, string> = {};
  const matches = (key: string, source: string) => same(feature(reference, key).shape, feature(candidate, source).shape);
  for (const key of cameraKeys) {
    if (remaining.has(key) && matches(key, key)) { mapping[key] = key; remaining.delete(key); }
  }
  for (const key of cameraKeys) {
    if (mapping[key]) continue;
    const source = [...remaining].find(source => matches(key, source));
    if (source) { mapping[key] = source; remaining.delete(source); }
  }
  return mapping;
}

/** Mirrors lifecycle/dataset_mixture.py; server validation remains authoritative. */
export function datasetCombinationIssue(reference: DatasetProfile, candidate: DatasetProfile, cameraKeys: string[], mapping = combinationCameraMapping(reference, candidate, cameraKeys)): string | null {
  if ([reference, candidate].some(profile => profile.source !== 'huggingface')) return 'Local training copies must be trained separately. Combined training supports Hub datasets.';
  if ([reference, candidate].some(profile => profile.format !== 'lerobot_v3')) return 'Combined training requires LeRobot v3 datasets. These dataset versions cannot be combined.';
  if ([reference, candidate].some(profile => !/^[0-9a-f]{40}$/i.test(profile.revision))) return 'Inspect a pinned, immutable revision before combining this dataset.';
  if (reference.repo_id === candidate.repo_id && reference.revision === candidate.revision) return 'This exact dataset revision is already selected.';
  if (reference.fps !== candidate.fps) return `Frame rates differ: ${candidate.fps} FPS here, ${reference.fps} FPS in the selected dataset.`;
  for (const [key, name] of [['action', 'Action'], ['observation.state', 'Robot state']]) {
    const a = feature(reference, key), b = feature(candidate, key);
    if (!Array.isArray(a.shape) || !a.shape.length || !Array.isArray(b.shape) || !b.shape.length || !same(a.shape, b.shape)) return `${name} vector dimensions do not match the selected dataset.`;
    if (!same(a.dtype, b.dtype)) return `${name} data types do not match the selected dataset.`;
    if (!same(a.names, b.names)) return `${name} field names or their order do not match the selected dataset.`;
  }
  if (!cameraKeys.length) return 'Select at least one camera before combining datasets.';
  if (imageKeys(candidate).length < cameraKeys.length) return `Needs ${cameraKeys.length} distinct matching cameras; this dataset has ${imageKeys(candidate).length}.`;
  if (!same(Object.keys(mapping).sort(), [...cameraKeys].sort()) || new Set(Object.values(mapping)).size !== cameraKeys.length) return 'Each selected camera needs a distinct matching view in this dataset.';
  for (const key of cameraKeys) {
    const mapped = feature(candidate, mapping[key]);
    if (!['image', 'video'].includes(String(mapped.dtype)) || !same(mapped.shape, feature(reference, key).shape)) return 'Camera dimensions do not match the selected cameras. Choose datasets with matching image sizes and channel counts.';
  }
  return null;
}
