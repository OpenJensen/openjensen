import { expect, test } from '@playwright/test';
import { runInNewContext } from 'node:vm';
import { getenv } from '../../apps/web/src/lib/environment';
import {
  readPublicPreference,
  writePublicPreference,
  themeBootstrapScript,
} from '../../apps/web/src/lib/public-preferences';
import { publicFormatLabel, publicMessage } from '../../apps/web/src/lib/public-display';
import { modelFormat } from '../../apps/web/src/lib/model-library';
import type { PolicyArtifact } from '../../apps/web/src/lib/api';

function storage(values: Record<string, string> = {}) {
  const items = new Map(Object.entries(values));
  return {
    items,
    getItem: (key: string) => items.get(key) ?? null,
    setItem: (key: string, value: string) => {
      items.set(key, value);
    },
  };
}

test('canonical server configuration wins over alpha aliases, including explicit empty values', () => {
  const settings = {
    JENSEN_BROWSER_PORT: '9123',
    FIREBIRD_BROWSER_PORT: '8765',
    JENSEN_PREFIX_SMOKE_EXPORT: '',
    FIREBIRD_PREFIX_SMOKE_EXPORT: '1',
  };
  expect(getenv('JENSEN_BROWSER_PORT', settings)).toBe('9123');
  expect(getenv('FIREBIRD_BROWSER_PORT', settings)).toBe('9123');
  expect(getenv('JENSEN_PREFIX_SMOKE_EXPORT', settings)).toBe('');
  expect(getenv('JENSEN_TEST_WEB_DIR', { FIREBIRD_TEST_WEB_DIR: '/alpha/export' })).toBe('/alpha/export');
  expect(getenv('JENSEN_BROWSER_PORT', {}, '8765')).toBe('8765');
  expect(
    getenv('NEXT_PUBLIC_BASE_PATH', { NEXT_PUBLIC_BASE_PATH: '/custom', JENSEN_NEXT_PUBLIC_BASE_PATH: '/wrong' }),
  ).toBe('/custom');
});

for (const name of ['theme', 'project', 'workflow.project-a'])
  test(`public ${name} preferences read alpha data and write only the canonical key`, () => {
    const legacy =
      name === 'theme' ? 'dark' : name === 'project' ? 'project-a' : JSON.stringify({ steps: 123, mode: 'libero' });
    const store = storage({
      [`firebird.${name}`]: legacy,
      'firebird:submission:v1:accepted': 'exact unresolved receipt',
    });
    expect(readPublicPreference(store, name)).toBe(legacy);
    writePublicPreference(store, name, 'new value');
    expect(readPublicPreference(store, name)).toBe('new value');
    expect(store.items.get(`firebird.${name}`)).toBe(legacy);
    expect(store.items.get('firebird:submission:v1:accepted')).toBe('exact unresolved receipt');
    writePublicPreference(store, name, '');
    expect(readPublicPreference(store, name)).toBe('');
  });

test('theme bootstrap matches canonical-first preference reads and survives denied storage', () => {
  for (const values of [
    { 'firebird.theme': 'dark' },
    { 'firebird.theme': 'dark', 'jensen.theme': 'light' },
    { 'firebird.theme': 'light', 'jensen.theme': 'dark' },
    { 'firebird.theme': 'dark', 'jensen.theme': '' },
    {},
  ]) {
    const localStorage = storage(values),
      document = { documentElement: { dataset: { theme: 'light' } } };
    runInNewContext(themeBootstrapScript, { localStorage, document });
    expect(document.documentElement.dataset.theme).toBe(
      readPublicPreference(localStorage, 'theme') === 'dark' ? 'dark' : 'light',
    );
  }
  const document = { documentElement: { dataset: { theme: 'dark' } } };
  runInNewContext(themeBootstrapScript, {
    localStorage: {
      getItem() {
        throw new Error('denied');
      },
    },
    document,
  });
  expect(document.documentElement.dataset.theme).toBe('light');
});

test('public labels preserve exact legacy wire and artifact values', () => {
  const reply = { message: 'Experimental Firebird prompt; not a reproduction of the upstream benchmark.' };
  expect(publicMessage(reply.message)).toBe(
    'Experimental OPEN JENSEN prompt; not a reproduction of the upstream benchmark.',
  );
  expect(reply.message).toContain('Experimental Firebird prompt;');
  expect(publicMessage('Historical Firebird results at the original source')).toBe(
    'Historical Firebird results at the original source',
  );
  const artifact = {
    format: 'native_quantized',
    metadata: { format: 'firebird_quant', precision: 'int8' },
  } as PolicyArtifact;
  expect(modelFormat(artifact)).toContain('OPEN JENSEN Quant');
  expect(artifact.metadata!.format).toBe('firebird_quant');
  expect(publicFormatLabel('safetensors')).toBe('safetensors');
});
