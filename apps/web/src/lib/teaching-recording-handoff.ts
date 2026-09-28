import type { ManagedPublishedCapture } from './managed-teaching';
import { recordingCatalog, recordingRecipe, type RecordingCatalog, type RecordingRecipe } from './recording-preparation';

const hash = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const id = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{32}$/.test(value);
const label = (value: unknown): value is string => typeof value === 'string' && value.length > 0 && value.length <= 256 && !/[\x00-\x1f\x7f]/.test(value);
const unavailable = () => new Error('The published teaching capture no longer matches this project’s recording catalog. Refresh and review it again; no episodes were substituted.');

/** Compare a reviewed publication to freshly read catalog metadata; no I/O or mutation. */
export function teachingRecordingRecipe(capture: ManagedPublishedCapture, project: string, value: RecordingCatalog, timeout: number): RecordingRecipe {
  const catalog = recordingCatalog(value);
  if (!capture || !label(project) || capture.project_id !== project || !label(capture.job_id) || !label(capture.profile_id) ||
      !hash(capture.profile_sha256) || !hash(capture.inventory_sha256) || !id(capture.session_id) || !hash(capture.session_sha256) ||
      catalog.configuration_sha256 !== capture.recording_configuration_sha256 || !Array.isArray(capture.episodes) ||
      capture.episodes.length < 1 || capture.episodes.length > 100 || capture.episodes.some(item => !item || !id(item.episode_id) || !hash(item.receipt_sha256)) || new Set(capture.episodes.map(item => item.episode_id)).size !== capture.episodes.length) throw unavailable();
  const source = catalog.captures.find(item => item.session_id === capture.session_id);
  if (!source || source.session_sha256 !== capture.session_sha256 || source.origin !== capture.origin || source.lineage_group !== capture.lineage_group ||
      source.episodes.length !== capture.episodes.length || capture.episodes.some(episode => {
        const current = source.episodes.find(item => item.episode_id === episode.episode_id);
        return !current || current.receipt_sha256 !== episode.receipt_sha256 || current.frames !== episode.frames ||
          current.termination !== episode.termination || current.outcome !== episode.outcome;
      })) throw unavailable();
  return recordingRecipe({ schema_version: 1, configuration_sha256: catalog.configuration_sha256, timeout_seconds: timeout,
    captures: [{ session_id: source.session_id, session_sha256: source.session_sha256,
      episodes: source.episodes.map(item => ({ episode_id: item.episode_id, receipt_sha256: item.receipt_sha256 })) }] });
}
