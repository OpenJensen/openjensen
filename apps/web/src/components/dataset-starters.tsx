import { datasetStarters, type DatasetStarter } from '@/lib/dataset-starters';

export function DatasetStarters({ selected, onSelect }: { selected: string; onSelect: (starter: DatasetStarter) => void }) {
  return <section className="starter-gallery" aria-labelledby="starters-title">
    <div className="panel-title"><h2 id="starters-title">Example datasets</h2></div>
    <div className="starter-grid">{datasetStarters.map(starter => <button type="button" key={starter.id} className={`starter-card${selected === starter.id ? ' selected' : ''}`} onClick={() => onSelect(starter)} aria-pressed={selected === starter.id}>
      <div className="starter-image"><img src={starter.poster} alt={`${starter.title} — source camera frame`} width="640" height="400" loading="lazy" /><span className="starter-camera-count">{starter.cameras} {starter.cameras === 1 ? 'camera' : 'cameras'}</span>{selected === starter.id && <span className="starter-selected" aria-label="Selected">✓</span>}</div>
      <div className="starter-copy"><h3>{starter.title}</h3><p className="starter-repo">{starter.repoId}</p><div className="starter-meta"><span>{starter.episodes} episodes</span><span>Use dataset <span aria-hidden="true">↗</span></span></div></div>
    </button>)}</div>
    <p className="starter-footnote">Select an example to inspect its cameras and episodes. Examples use pinned dataset versions.</p>
  </section>;
}
