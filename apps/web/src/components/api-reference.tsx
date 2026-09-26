'use client';

import { useEffect, useState } from 'react';
import { WorkspaceShell } from '@/components/workspace-shell';
import { Icon } from '@/components/icon';
import { apiOrigin, apiReferenceUrl, openApiUrl } from '@/lib/api';
import { publicPath } from '@/lib/base-path';
import { curlExample, endpointsFor, filterEndpoints, httpMethods, parseDocument, referenceName, schemaType, type Endpoint, type Media, type OpenApiDocument, type Schema } from '@/lib/openapi';

function SchemaLink({ schema }: { schema: Schema }) {
  if (schema.$ref?.startsWith('#/components/schemas/')) {
    const name = referenceName(schema.$ref);
    const id = `schema-${encodeURIComponent(name)}`;
    return <a href={`#${id}`} onClick={() => {
      const target = window.document.getElementById(id);
      if (target instanceof HTMLDetailsElement) target.open = true;
    }}><code>{name}</code></a>;
  }
  if (schema.type === 'array' && schema.items) return <>array&lt;<SchemaLink schema={schema.items} />&gt;</>;
  const alternatives = schema.anyOf ?? schema.oneOf ?? schema.allOf;
  if (alternatives) return <>{alternatives.map((part, index) => <span key={index}>{index > 0 && (schema.allOf ? ' & ' : ' | ')}<SchemaLink schema={part} /></span>)}</>;
  return <code>{schemaType(schema)}</code>;
}

function SchemaFields({ schema }: { schema: Schema }) {
  if (!schema.properties) return <p className="reference-type"><SchemaLink schema={schema} /></p>;
  return <div className="table-scroll"><table className="feature-table reference-table">
    <caption className="visually-hidden">Schema fields</caption>
    <thead><tr><th scope="col">Field</th><th scope="col">Type & constraints</th></tr></thead>
    <tbody>{Object.entries(schema.properties).map(([name, field]) => <tr key={name}>
      <th scope="row"><code>{name}</code><span className="field-requirement">{schema.required?.includes(name) ? 'required' : 'optional'}</span></th>
      <td><SchemaLink schema={field} />{field.description && <p>{field.description}</p>}
        {field.default !== undefined && <p>Default: <code>{JSON.stringify(field.default)}</code></p>}
        <SchemaConstraints schema={field} />
      </td>
    </tr>)}</tbody>
  </table></div>;
}

function SchemaConstraints({ schema }: { schema: Schema }) {
  return <>{(['minLength', 'maxLength', 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'pattern', 'format'] as const).map(key => schema[key] !== undefined && <p key={key}>{key}: <code>{JSON.stringify(schema[key])}</code></p>)}
    {(schema.anyOf ?? schema.oneOf ?? schema.allOf)?.map((part, index) => <SchemaConstraints schema={part} key={index} />)}
  </>;
}

function MediaSchema({ content }: { content?: Record<string, Media> }) {
  return content ? <>{Object.entries(content).map(([type, media]) => <div key={type} className="reference-media">
    <span className="reference-content-type">{type}</span>
    {media.schema && <SchemaFields schema={media.schema} />}
    {media.example !== undefined && <pre className="reference-code"><code>{JSON.stringify(media.example, null, 2)}</code></pre>}
  </div>)}</> : <p className="reference-hint">No response body declared.</p>;
}

function RequestExample({ endpoint, origin }: { endpoint: Endpoint; origin: string }) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const example = curlExample(endpoint, origin);
  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(timer);
  }, [copied]);
  async function copy() {
    try {
      await navigator.clipboard.writeText(example);
      setCopied(true);
      setCopyError(false);
    } catch { setCopyError(true); }
  }
  return <div className="reference-example">
    <div className="reference-example-heading"><span>cURL</span><button className="text-button" type="button" onClick={() => void copy()} aria-label={`Copy ${endpoint.method} ${endpoint.path} example`}>{copied ? 'Copied' : 'Copy example'}</button></div>
    <pre className="reference-code"><code>{example}</code></pre>
    {endpoint.parameters.length > 0 && <p className="reference-hint">Replace path placeholders and supply the parameters listed above before running.</p>}
    {endpoint.operation.requestBody && <p className="reference-hint">Create <code>request.json</code> with the fields above.</p>}
    <span role="status" className={copyError ? 'reference-hint' : 'visually-hidden'}>{copyError ? 'Copy unavailable. Select the example text to copy it.' : copied ? 'Example copied to clipboard.' : ''}</span>
  </div>;
}

function EndpointCard({ endpoint, origin }: { endpoint: Endpoint; origin: string }) {
  const { operation } = endpoint;
  return <details className="reference-endpoint" id={endpoint.id}>
    <summary>
      <span className={`method-badge method-${endpoint.method.toLowerCase()}`}>{endpoint.method}</span>{' '}
      <code className="endpoint-path">{endpoint.path}</code>{' '}
      <span className="endpoint-summary">{operation.summary ?? operation.operationId ?? 'Endpoint'}{operation.deprecated && ' · Deprecated'}</span>
      <span className="endpoint-chevron" aria-hidden="true">⌄</span>
    </summary>
    <div className="endpoint-body">
      <div className="endpoint-description"><h3>{operation.summary ?? 'Endpoint details'}</h3><a className="text-link" href={`#${endpoint.id}`}>Permalink</a></div>
      {operation.description && <p className="reference-hint">{operation.description}</p>}
      <div className="endpoint-detail-grid"><div>
        {endpoint.parameters.length > 0 && <section className="endpoint-section"><h4>Parameters</h4><div className="table-scroll"><table className="feature-table reference-table">
          <caption className="visually-hidden">Endpoint parameters</caption><thead><tr><th scope="col">Parameter</th><th scope="col">Type & location</th></tr></thead>
          <tbody>{endpoint.parameters.map(parameter => <tr key={`${parameter.in}-${parameter.name}`}><th scope="row"><code>{parameter.name}</code><span className="field-requirement">{parameter.required ? 'required' : 'optional'}</span></th><td>{parameter.schema && <SchemaLink schema={parameter.schema} />}<p>{parameter.in}</p>{parameter.description && <p>{parameter.description}</p>}</td></tr>)}</tbody>
        </table></div></section>}
        {operation.requestBody && <section className="endpoint-section"><h4>Request body <span>{operation.requestBody.required ? 'required' : 'optional'}</span></h4>{operation.requestBody.description && <p className="reference-hint">{operation.requestBody.description}</p>}<MediaSchema content={operation.requestBody.content} /></section>}
        <section className="endpoint-section"><h4>Responses</h4><div className="reference-responses">{Object.entries(operation.responses ?? {}).map(([status, response]) => <div key={status} className="reference-response"><div className="response-heading"><code>{status}</code><span>{response.description}</span></div><MediaSchema content={response.content} /></div>)}</div></section>
      </div><RequestExample endpoint={endpoint} origin={origin} /></div>
    </div>
  </details>;
}

export function ApiReference() {
  const [document, setDocument] = useState<OpenApiDocument | null>(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  const [origin, setOrigin] = useState(apiOrigin);
  const [search, setSearch] = useState('');
  const [method, setMethod] = useState('ALL');

  useEffect(() => {
    setOrigin(new URL(apiOrigin || '/', window.location.origin).href.replace(/\/$/, ''));
    const controller = new AbortController();
    let active = true;
    const timeout = setTimeout(() => controller.abort(), 15000);
    setError('');
    setDocument(null);
    void fetch(openApiUrl, { signal: controller.signal, cache: 'no-store' }).then(async response => {
      if (!response.ok) throw new Error(`The API schema could not be loaded (${response.status}).`);
      return parseDocument(await response.json());
    }).then(schema => { if (active) setDocument(schema); }).catch((cause: unknown) => {
      if (active) setError(cause instanceof Error && cause.name !== 'AbortError' ? cause.message : 'The API schema request timed out. Check that the API is running.');
    }).finally(() => clearTimeout(timeout));
    return () => { active = false; clearTimeout(timeout); controller.abort(); };
  }, [attempt]);

  // Native anchors remain shareable even though endpoints arrive asynchronously.
  useEffect(() => {
    function revealAnchor() {
      let id: string;
      try { id = decodeURIComponent(window.location.hash.slice(1)); } catch { return; }
      // Endpoint IDs contain encoded path segments; try the literal ID first.
      const target = window.document.getElementById(window.location.hash.slice(1)) ?? window.document.getElementById(id);
      if (target instanceof HTMLDetailsElement) target.open = true;
      if (target) target.scrollIntoView({ block: 'start' });
    }
    revealAnchor();
    window.addEventListener('hashchange', revealAnchor);
    return () => window.removeEventListener('hashchange', revealAnchor);
  }, [document]);

  const endpoints = document ? endpointsFor(document) : [];
  const filtered = filterEndpoints(endpoints, search, method);
  const schemas = Object.entries(document?.components?.schemas ?? {});
  const navigation = <nav className="reference-navigation" aria-label="API navigation">
    <p className="sidebar-section-label">Workspace</p>
    <ul className="stage-list"><li><a className="stage-button" href={publicPath('/')}><Icon name="database" size={19} /><span>Dataset workspace</span></a></li><li><a className="stage-button selected" href={apiReferenceUrl} aria-current="page"><Icon name="book" size={19} /><span>API reference</span></a></li></ul>
    <div className="reference-nav-section"><p className="sidebar-section-label">On this page</p><ul className="stage-list"><li><a className="stage-button" href="#overview">Overview</a></li><li><a className="stage-button" href="#endpoints">Endpoints <small>{endpoints.length || '—'}</small></a></li><li><a className="stage-button" href="#schemas">Data models <small>{schemas.length || '—'}</small></a></li></ul></div>
  </nav>;

  return <WorkspaceShell navigation={navigation} sidebarLabel="Reference navigation" skipLabel="Skip to API reference" contentClassName="reference-content" breadcrumb={<><Icon name="book" size={18} /><strong>API reference</strong><span className="breadcrumb-divider">/</span><span className="breadcrumb-project">Developer tools</span></>}>
    <section id="overview" className="reference-overview" aria-labelledby="reference-heading">
      <div className="page-heading"><div><h1 id="reference-heading">API reference</h1></div><a className="secondary-button reference-schema-link" href={openApiUrl}>OpenAPI JSON <Icon name="external" size={14} /></a></div>
      <div className="reference-overview-grid"><div><span className="eyebrow">Base URL</span><code>{origin}/api/v1</code></div><div><span className="eyebrow">API version</span><strong>{document ? `v${document.info.version}` : '—'}</strong></div><div><span className="eyebrow">Contract</span><strong>{document ? `OpenAPI ${document.openapi}` : 'Loading…'}</strong></div></div>
    </section>

    {!document && !error && <div className="reference-state" role="status"><span className="spinner" /><h2>Loading API reference…</h2></div>}
    {error && <div className="reference-state"><span className="empty-icon"><Icon name="book" size={24} /></span><h2>API reference unavailable</h2><p role="alert">{error}</p><button className="secondary-button" type="button" onClick={() => setAttempt(value => value + 1)}>Retry loading schema <Icon name="arrow" size={15} /></button></div>}
    {document && <>
      <section id="endpoints" className="reference-section" aria-labelledby="endpoints-heading">
        <div className="section-tabs"><h2 className="section-tab" id="endpoints-heading">Endpoints</h2><span className="section-note">{endpoints.length} operations · JSON</span></div>
        <div className="reference-filters"><div className="reference-search"><Icon name="book" size={16} /><label className="visually-hidden" htmlFor="endpoint-search">Search endpoints</label><input id="endpoint-search" type="search" placeholder="Search endpoints, methods, or descriptions…" value={search} onChange={event => setSearch(event.target.value)} /></div><label className="visually-hidden" htmlFor="method-filter">HTTP method</label><select id="method-filter" value={method} onChange={event => setMethod(event.target.value)}><option value="ALL">All methods</option>{httpMethods.filter(value => endpoints.some(endpoint => endpoint.method === value.toUpperCase())).map(value => <option key={value} value={value.toUpperCase()}>{value.toUpperCase()}</option>)}</select></div>
        <p className="reference-result-count" role="status">Showing {filtered.length} of {endpoints.length} endpoints</p>
        <div className="reference-endpoints">{filtered.map(endpoint => <EndpointCard key={endpoint.id} endpoint={endpoint} origin={origin} />)}</div>
        {!filtered.length && <div className="reference-state"><h3>No endpoints found</h3><button className="secondary-button" type="button" onClick={() => { setSearch(''); setMethod('ALL'); }}>Clear filters</button></div>}
      </section>
      <section id="schemas" className="reference-section" aria-labelledby="schemas-heading"><div className="section-tabs"><h2 className="section-tab" id="schemas-heading">Data models</h2><span className="section-note">{schemas.length} schemas</span></div><div className="reference-models">{schemas.map(([name, schema]) => <details className="reference-model" id={`schema-${encodeURIComponent(name)}`} key={name}><summary><Icon name="layers" size={17} /><code>{name}</code><span>{Object.keys(schema.properties ?? {}).length} fields</span><span className="endpoint-chevron" aria-hidden="true">⌄</span></summary><div className="reference-model-body">{schema.description && <p className="reference-hint">{schema.description}</p>}<SchemaFields schema={schema} /><details className="reference-raw"><summary>Full JSON schema</summary><pre className="reference-code"><code>{JSON.stringify(schema, null, 2)}</code></pre></details></div></details>)}</div></section>
      <footer className="workspace-footer"><a className="text-link" href={publicPath('/')}>Back to workspace <Icon name="arrow" size={13} /></a></footer>
    </>}
  </WorkspaceShell>;
}
