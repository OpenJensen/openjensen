import { datasetStarters, type DatasetStarter } from '@/lib/dataset-starters';

export function DatasetStarters({ selected, onSelect }: { selected: string; onSelect: (starter: DatasetStarter) => void }) {
  return <section className="starter-gallery" aria-labelledby="starters-title">
    <div className="panel-title"><h2 id="starters-title">Explore a real dataset</h2><p>Choose a starting point, then inspect and preview.</p></div>
    <div className="starter-grid">{datasetStarters.map(starter => <button type="button" key={starter.id} className={`starter-card${selected === starter.id ? ' selected' : ''}`} onClick={() => onSelect(starter)} aria-pressed={selected === starter.id}>
      <div className="starter-image"><img src={starter.poster} alt={`${starter.title} — source camera frame`} width="640" height="400" /><span className="starter-camera-count">{starter.cameras} {starter.cameras === 1 ? 'camera' : 'cameras'}</span>{selected === starter.id && <span className="starter-selected" aria-label="Selected">✓</span>}</div>
      <div className="starter-copy"><h3>{starter.title}</h3><p className="starter-repo">{starter.repoId}</p><p>{starter.description}</p><div className="starter-meta"><span>{starter.episodes} episodes</span><span>Use dataset <span aria-hidden="true">↗</span></span></div></div>
    </button>)}</div>
    <p className="starter-footnote">Preview frames from the original datasets. Each example uses a fixed source revision.</p>
  </section>;
}
