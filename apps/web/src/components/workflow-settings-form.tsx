import type { Preferences } from '@/lib/workflow-preferences';

/** Project defaults for compression, evaluation and fine-tuning used by later workflow runs. */
export function WorkflowSettingsForm({
  preferences,
  ready,
  language,
  update,
  onSuite,
}: {
  preferences: Preferences;
  ready: boolean;
  language: Preferences['precision'];
  update: <K extends keyof Preferences>(key: K, value: Preferences[K]) => void;
  onSuite: (suite: Preferences['suite']) => void;
}) {
  return (
    <fieldset className="workflow-fields workflow-controls" disabled={!ready}>
      <legend className="visually-hidden">Project workflow settings</legend>
      <h2>Compression defaults</h2>
      <label>
        Quantization recipe
        <select value={language} onChange={(e) => update('precision', e.target.value as Preferences['precision'])}>
          <option value="Q4_0">LM Q4 (experimental)</option>
          <option value="Q8_0">LM Q8</option>
        </select>
      </label>
      <label className="workflow-check">
        <input
          type="checkbox"
          checked={preferences.compareQ4}
          onChange={(e) => update('compareQ4', e.target.checked)}
        />
        Compare Q8 and Q4 (experimental)
      </label>
      <label className="workflow-check">
        <input type="checkbox" checked={preferences.vision} onChange={(e) => update('vision', e.target.checked)} />
        Also quantize vision to Q8 (experimental)
      </label>
      <h2>Evaluation</h2>
      <label>
        Task suite
        <select
          value={preferences.suite}
          onChange={(e) => {
            if (ready) onSuite(e.target.value as Preferences['suite']);
          }}
        >
          <option value="libero_object">LIBERO Object (legacy policy)</option>
          <option value="libero_spatial">LIBERO Spatial (pinned SmolVLA policy)</option>
        </select>
      </label>
      {preferences.suite === 'libero_spatial' && (
        <>
          <label>
            Spatial task IDs
            <input value={preferences.taskIds} onChange={(e) => update('taskIds', e.target.value)} />
          </label>
          <label>
            Approved parity profile
            <input
              value={preferences.parityProfile}
              onChange={(e) => update('parityProfile', e.target.value)}
              placeholder="Name your reviewed tolerance profile"
            />
          </label>
          <label>
            Maximum action RMSE
            <input
              type="number"
              min="0"
              step="any"
              value={preferences.parityRmse}
              onChange={(e) => update('parityRmse', e.target.value)}
            />
          </label>
          <label>
            Maximum absolute action error
            <input
              type="number"
              min="0"
              step="any"
              value={preferences.parityMaxError}
              onChange={(e) => update('parityMaxError', e.target.value)}
            />
          </label>
        </>
      )}
      <label>
        Protocol
        <select
          value={preferences.mode}
          disabled={preferences.suite === 'libero_spatial'}
          onChange={(e) => update('mode', e.target.value as Preferences['mode'])}
        >
          <option value="engine">Engine diagnostics</option>
          <option value="libero">Paired LIBERO episodes</option>
        </select>
      </label>
      <label>
        Timed predictions
        <input
          type="number"
          min="2"
          max="1000"
          value={preferences.repetitions}
          onChange={(e) => update('repetitions', Number(e.target.value))}
        />
      </label>
      <label>
        Warmup predictions
        <input
          type="number"
          min="1"
          max="100"
          value={preferences.warmup}
          onChange={(e) => update('warmup', Number(e.target.value))}
        />
      </label>
      {preferences.mode === 'libero' && (
        <>
          {preferences.suite === 'libero_object' && (
            <label>
              LIBERO object task
              <input
                type="number"
                min="0"
                max="9"
                value={preferences.task}
                onChange={(e) => update('task', Number(e.target.value))}
              />
            </label>
          )}
          <label>
            Search initial states
            <input value={preferences.searchStates} onChange={(e) => update('searchStates', e.target.value)} />
          </label>
          <label>
            Unused final initial states
            <input value={preferences.finalStates} onChange={(e) => update('finalStates', e.target.value)} />
          </label>
          <label>
            Episode step limit
            <input
              type="number"
              min="1"
              max="500"
              value={preferences.steps}
              disabled={preferences.suite === 'libero_spatial'}
              onChange={(e) => update('steps', Number(e.target.value))}
            />
          </label>
          <label className="workflow-check">
            <input type="checkbox" checked={preferences.select} onChange={(e) => update('select', e.target.checked)} />
            Select the smallest candidate that meets my limits
          </label>
          {preferences.select && (
            <>
              <label>
                Minimum task success (%)
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={preferences.minSuccess}
                  onChange={(e) => update('minSuccess', Number(e.target.value))}
                />
              </label>
              <label>
                Maximum p95 latency (ms)
                <input
                  type="number"
                  min="1"
                  value={preferences.maxLatency}
                  onChange={(e) => update('maxLatency', Number(e.target.value))}
                />
              </label>
              <label>
                Maximum measured target memory (MiB)
                <input
                  type="number"
                  min="1"
                  value={preferences.maxMemory}
                  onChange={(e) => update('maxMemory', Number(e.target.value))}
                />
              </label>
            </>
          )}
        </>
      )}
      <h2>Fine-tuning</h2>
      <label>
        Optimizer steps
        <input
          type="number"
          min="2"
          value={preferences.trainingSteps}
          onChange={(e) => update('trainingSteps', Number(e.target.value))}
        />
      </label>
    </fieldset>
  );
}
