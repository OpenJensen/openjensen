import { Icon } from '@/components/icon';

export function ModeCard({ title, detail, icon, selected, disabled, onClick }: { title: string; detail: string; icon?: Parameters<typeof Icon>[0]['name']; selected: boolean; disabled: boolean; onClick: () => void }) {
  return <button type="button" className={`mode-card${icon ? '' : ' mode-card-plain'}${selected ? ' selected' : ''}`} aria-label={title} aria-pressed={selected} disabled={disabled} onClick={onClick}>
    <span className="mode-card-top">{icon ? <span className="mode-card-icon"><Icon name={icon} size={23} /></span> : <strong>{title}</strong>}<span className="mode-card-check">{selected && <Icon name="check" size={13} />}</span></span>
    {icon && <strong>{title}</strong>}<span className="mode-card-detail">{detail}</span>
  </button>;
}

export function ErrorNotice({ error }: { error: Error | null }) {
  return error ? <p className="error-notice" role="alert">{error.message}</p> : null;
}

export function displayDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}
