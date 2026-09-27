import { expect, test } from '@playwright/test';
import {
  curlExample, endpointsFor, filterEndpoints, parseDocument, referenceName, schemaType, shellQuote,
  type OpenApiDocument,
} from '../../apps/web/src/lib/openapi';

const document: OpenApiDocument = {
  openapi: '3.1.0',
  info: { title: 'Regression fixture', version: '1.0' },
  paths: {
    '/api/v1/widgets/{id}': {
      parameters: [{ name: 'id', in: 'path', required: true, schema: { type: 'string' } }],
      get: {
        summary: 'Read a widget', operationId: 'read_widget', tags: ['Inventory'],
        description: 'Includes metadata and ownership.',
        responses: { '200': { description: 'Found' } },
      },
      post: { summary: 'Update widget', requestBody: { content: { 'application/json': { schema: { type: 'object' } } } } },
    },
  },
};

test('accepts OpenAPI 3 documents and rejects unusable schema envelopes', () => {
  expect(parseDocument(document)).toEqual(document);
  for (const invalid of [null, [], 'schema', {}, { ...document, openapi: '2.0' },
    { ...document, info: null }, { ...document, info: { title: 'Missing version' } },
    { ...document, paths: [] }, { ...document, paths: null }]) {
    expect(() => parseDocument(invalid)).toThrow(/invalid OpenAPI document/);
  }
});

test('discovers all HTTP methods while ignoring path-level metadata', () => {
  const methods = ['get', 'post', 'put', 'patch', 'delete', 'options', 'head', 'trace'];
  const schema = structuredClone(document);
  schema.paths['/new'] = { summary: 'Metadata', servers: [], ...Object.fromEntries(methods.map(method => [method, {}])) };
  const endpoints = endpointsFor(schema).filter(endpoint => endpoint.path === '/new');
  expect(endpoints.map(endpoint => endpoint.method)).toEqual(methods.map(method => method.toUpperCase()));
  expect(new Set(endpoints.map(endpoint => endpoint.id)).size).toBe(methods.length);
});

test('inherits path parameters and overrides them by both location and name', () => {
  const schema = structuredClone(document);
  schema.paths['/api/v1/widgets/{id}'].get = {
    parameters: [
      { name: 'id', in: 'path', required: true, description: 'Operation override' },
      { name: 'id', in: 'query', required: false },
    ],
  };
  const [get, post] = endpointsFor(schema);
  expect(get.parameters).toEqual([
    { name: 'id', in: 'path', required: true, description: 'Operation override' },
    { name: 'id', in: 'query', required: false },
  ]);
  expect(post.parameters).toEqual(document.paths['/api/v1/widgets/{id}'].parameters);
});

test('searches all endpoint metadata with case-insensitive AND terms and a method filter', () => {
  const endpoints = endpointsFor(document);
  expect(filterEndpoints(endpoints, '  WIDGET    metadata  ', 'ALL')).toEqual([endpoints[0]]);
  expect(filterEndpoints(endpoints, 'inventory read_widget', 'GET')).toEqual([endpoints[0]]);
  expect(filterEndpoints(endpoints, '/api/v1/widgets', 'POST')).toEqual([endpoints[1]]);
  expect(filterEndpoints(endpoints, 'ownership', 'POST')).toEqual([]);
  expect(filterEndpoints(endpoints, 'nonexistent', 'ALL')).toEqual([]);
  expect(filterEndpoints(endpoints, '  ', 'ALL')).toEqual(endpoints);
});

test('formats referenced, nullable, array, enum, constant, and composite model types', () => {
  expect(referenceName('#/components/schemas/Path~1Name~0')).toBe('Path/Name~');
  expect(schemaType({ $ref: '#/components/schemas/Project' })).toBe('Project');
  expect(schemaType({ anyOf: [{ type: 'string' }, { type: 'null' }] })).toBe('string | null');
  expect(schemaType({ oneOf: [{ type: 'number' }, { type: 'boolean' }] })).toBe('number | boolean');
  expect(schemaType({ type: 'array', items: { $ref: '#/components/schemas/Job' } })).toBe('array<Job>');
  expect(schemaType({ enum: ['queued', 'running', null] })).toBe('"queued" | "running" | null');
  expect(schemaType({ const: false })).toBe('false');
  expect(schemaType({ type: ['integer', 'null'] })).toBe('integer | null');
  expect(schemaType({ allOf: [{ $ref: '#/components/schemas/Base' }, { type: 'object' }] })).toBe('Base & object');
  expect(schemaType({})).toBe('any');
});

test('GET examples use the active origin and preserve substitution placeholders', () => {
  const command = curlExample(endpointsFor(document)[0], 'http://localhost:8765');
  expect(command).toContain("curl --request GET 'http://localhost:8765/api/v1/widgets/{id}'");
  expect(command).toContain("--header 'Accept: application/json'");
  expect(command).not.toContain('--data');
  expect(command).not.toContain('Content-Type');
});

test('JSON examples use schema-authored request files instead of invented payloads', () => {
  const command = curlExample(endpointsFor(document)[1], 'https://api.example.test');
  expect(command).toContain("curl --request POST 'https://api.example.test/api/v1/widgets/{id}'");
  expect(command).toContain("--header 'Content-Type: application/json'");
  expect(command).toContain('--data @request.json');
  expect(command).not.toContain('{"');
});

test('shell quoting preserves apostrophes and blocks command substitution', () => {
  expect(shellQuote("a'b")).toBe("'a'\"'\"'b'");
  expect(shellQuote('$(touch sentinel); `id`')).toBe("'$(touch sentinel); `id`'");
});


test('cURL examples preserve a deployed API prefix', () => {
  const command = curlExample(endpointsFor(document)[0], 'https://example.test/firebird');
  expect(command).toContain("'https://example.test/firebird/api/v1/widgets/{id}'");
});
