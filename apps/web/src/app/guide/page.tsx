import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { WorkspaceShell } from '@/components/workspace-shell';
import { Icon } from '@/components/icon';
import { publicPath } from '@/lib/base-path';
import { guideSectionId } from '@/lib/guide-section';
import './guide.css';

export const metadata: Metadata = { title: 'Open Jensen · Workspace guide', description: 'A short guide to every part of the robotics workspace.' };

// Read the same concise guide that ships in the repository. No separate copy to maintain.
async function readGuide() {
  const source = await readFile(path.resolve(process.cwd(), '../../docs/workspace-guide.md'), 'utf8');
  const [intro, ...sections] = source.split(/^## /m);
  return {
    intro: intro.replace(/^# [^\n]+\n+/, '').trim(),
    sections: sections.map(section => {
      const split = section.indexOf('\n');
      const title = section.slice(0, split).trim();
      return { title, id: guideSectionId(title), body: section.slice(split + 1).trim() };
    }),
  };
}

function guideHref(href: string) {
  const local = href.match(/^http:\/\/127\.0\.0\.1:8000(\/(?:guide|docs)\/.*)$/);
  if (local) return publicPath(local[1]);
  if (href.startsWith('#')) return href;
  if (href.startsWith('/') && !href.startsWith('//')) return publicPath(href);
  if (/^https?:\/\//.test(href)) return href;
  if (/^[a-z]+:/i.test(href) || href.startsWith('//')) return '#';
  const target = path.posix.normalize(`docs/${href}`);
  if (target.startsWith('../')) return '#';
  return `https://github.com/OpenJensen/openjensen/blob/main/${target}`;
}

// The guide uses paragraphs, links, bold labels and inline code only.
function inline(text: string): ReactNode[] {
  const tokens = /\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*|`([^`]+)`/g;
  const output: ReactNode[] = [];
  let cursor = 0;
  for (const match of text.matchAll(tokens)) {
    const index = match.index!;
    if (index > cursor) output.push(text.slice(cursor, index));
    if (match[1]) {
      const href = guideHref(match[2]);
      const external = /^https?:/.test(href);
      output.push(<a key={index} href={href} {...(external ? { target: '_blank', rel: 'noreferrer' } : {})}>{inline(match[1])}</a>);
    } else if (match[3]) output.push(<strong key={index}>{match[3]}</strong>);
    else output.push(<code key={index}>{match[4]}</code>);
    cursor = index + match[0].length;
  }
  if (cursor < text.length) output.push(text.slice(cursor));
  return output;
}

function Paragraphs({ text }: { text: string }) {
  return text.split(/\n\s*\n/).filter(Boolean).map((paragraph, index) => <p key={index}>{inline(paragraph.replace(/\n/g, ' '))}</p>);
}

export default async function GuidePage() {
  const guide = await readGuide();
  return <WorkspaceShell showGuideShortcut={false} contentClassName="guide-page" skipLabel="Skip to guide" sidebarLabel="Guide navigation"
    breadcrumb={<><Icon name="book" size={18} /><strong>Guide</strong></>}
    navigation={<nav className="guide-navigation" aria-label="Guide sections">
      <a className="guide-back" href={publicPath('/dashboard/')}><Icon name="arrow" size={16} />Back to workspace</a>
      <ul>{guide.sections.map(section => <li key={section.id}><a href={`#${section.id}`}>{section.title}</a></li>)}</ul>
    </nav>}>
    <div className="page-heading"><h1>Workspace guide</h1></div>
    <div className="guide-intro"><Paragraphs text={guide.intro} /></div>
    <div className="guide-sections">{guide.sections.map(section => <section key={section.id} id={section.id} aria-labelledby={`${section.id}-title`}>
      <h2 id={`${section.id}-title`}>{section.title}</h2><Paragraphs text={section.body} />
    </section>)}</div>
  </WorkspaceShell>;
}
