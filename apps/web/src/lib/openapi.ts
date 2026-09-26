/** A small, read-only OpenAPI view model. The running API remains the source of truth. */
export type Schema = {
  $ref?: string;
  title?: string;
  description?: string;
  type?: string | string[];
  properties?: Record<string, Schema>;
  required?: string[];
  items?: Schema;
  anyOf?: Schema[];
  oneOf?: Schema[];
  allOf?: Schema[];
  enum?: unknown[];
  const?: unknown;
  default?: unknown;
  [key: string]: unknown;
};
export type Parameter = { name: string; in: string; required?: boolean; description?: string; schema?: Schema };
export type Media = { schema?: Schema; example?: unknown };
export type Operation = {
  summary?: string;
  description?: string;
  operationId?: string;
  tags?: string[];
  deprecated?: boolean;
  parameters?: Parameter[];
  requestBody?: { required?: boolean; description?: string; content?: Record<string, Media> };
  responses?: Record<string, { description?: string; content?: Record<string, Media> }>;
};
export type OpenApiDocument = {
  openapi: string;
  info: { title: string; version: string; description?: string };
  paths: Record<string, { parameters?: Parameter[] } & Record<string, unknown>>;
  components?: { schemas?: Record<string, Schema> };
};
export const httpMethods = ['get', 'post', 'put', 'patch', 'delete', 'options', 'head', 'trace'] as const;
export type Endpoint = { id: string; method: string; path: string; operation: Operation; parameters: Parameter[] };

export function parseDocument(value: unknown): OpenApiDocument {
  if (!value || typeof value !== 'object') throw new Error('The API returned an invalid OpenAPI document.');
  const doc = value as OpenApiDocument;
  if (typeof doc.openapi !== 'string' || !doc.openapi.startsWith('3.') ||
      !doc.info || typeof doc.info.title !== 'string' || typeof doc.info.version !== 'string' ||
      !doc.paths || typeof doc.paths !== 'object' || Array.isArray(doc.paths)) {
    throw new Error('The API returned an invalid OpenAPI document.');
  }
  return doc;
}

export function endpointsFor(doc: OpenApiDocument): Endpoint[] {
  return Object.entries(doc.paths).flatMap(([path, item]) => httpMethods.flatMap(method => {
    const operation = item[method] as Operation | undefined;
    if (!operation || typeof operation !== 'object') return [];
    const parameters = new Map<string, Parameter>();
    for (const parameter of [...(item.parameters ?? []), ...(operation.parameters ?? [])]) {
      parameters.set(`${parameter.in}:${parameter.name}`, parameter);
    }
    return [{ id: `endpoint-${method}-${encodeURIComponent(path)}`, method: method.toUpperCase(), path, operation, parameters: [...parameters.values()] }];
  }));
}

export function filterEndpoints(endpoints: Endpoint[], search: string, method: string): Endpoint[] {
  const terms = search.trim().toLowerCase().split(/\s+/).filter(Boolean);
  return endpoints.filter(endpoint => (method === 'ALL' || endpoint.method === method) &&
    terms.every(term => `${endpoint.method} ${endpoint.path} ${endpoint.operation.summary ?? ''} ${endpoint.operation.description ?? ''} ${endpoint.operation.operationId ?? ''} ${endpoint.operation.tags?.join(' ') ?? ''}`.toLowerCase().includes(term)));
}

export function referenceName(ref: string): string {
  return ref.split('/').at(-1)?.replace(/~1/g, '/').replace(/~0/g, '~') ?? ref;
}

export function schemaType(schema: Schema): string {
  if (schema.$ref) return referenceName(schema.$ref);
  const alternatives = schema.anyOf ?? schema.oneOf;
  if (alternatives) return alternatives.map(schemaType).join(' | ');
  if (schema.allOf) return schema.allOf.map(schemaType).join(' & ');
  if (schema.const !== undefined) return JSON.stringify(schema.const);
  if (schema.enum) return schema.enum.map(value => JSON.stringify(value)).join(' | ');
  if (schema.type === 'array') return `array<${schemaType(schema.items ?? {})}>`;
  return Array.isArray(schema.type) ? schema.type.join(' | ') : schema.type ?? 'any';
}

export function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'"'"'`)}'`;
}

export function curlExample(endpoint: Endpoint, origin: string): string {
  const url = new URL(endpoint.path, origin).href.replace(/%7B/gi, '{').replace(/%7D/gi, '}');
  const lines = [`curl --request ${endpoint.method} ${shellQuote(url)}`, `  --header 'Accept: application/json'`];
  // The reference never fabricates request bodies: payloads come from a local file
  // written against the schema shown alongside the example.
  if (endpoint.operation.requestBody?.content?.['application/json']) {
    lines.push(`  --header 'Content-Type: application/json'`, `  --data @request.json`);
  }
  return lines.join(' \\\n');
}
