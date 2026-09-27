// Pure bundled-screen controller tests. No browser, HTTP server or native child is started.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../frontend/connection.js'), 'utf8');
const state = (extra = {}) => ({ mode: 'idle', address: null, payload_available: false, cleanup_unknown: false, message: 'No verified payload', ...extra });
const tick = () => new Promise(resolve => setImmediate(resolve));
async function screen(initial, respond = async () => { throw new Error('unexpected invocation'); }) {
  let status = initial;
  const calls = [], navigations = [], elements = new Map();
  for (const id of ['connect', 'heading', 'message', 'owned-message', 'confirm', 'start', 'stop', 'restart', 'open-owned']) {
    elements.set(`#${id}`, { disabled: ['confirm', 'start', 'stop', 'restart'].includes(id), hidden: id === 'open-owned', checked: false, textContent: '', handlers: {}, addEventListener(event, fn) { this.handlers[event] = fn; } });
  }
  vm.runInNewContext(source, { document: { querySelector: id => elements.get(id) }, window: { location: { replace: value => navigations.push(value) }, __TAURI__: { core: { invoke: async (command, args) => {
    calls.push([command, args]);
    if (command === 'desktop_status') return status;
    return respond(command, args, value => { status = value; });
  } } } } });
  await tick();
  return { calls, navigations, get: id => elements.get(`#${id}`), click: async id => { await elements.get(`#${id}`).handlers.click(); await tick(); } };
}

test('missing verified payload remains disabled, without automatic Start', async () => {
  const ui = await screen(state());
  assert.equal(ui.get('start').disabled, true);
  assert.equal(ui.get('confirm').disabled, true);
  assert.equal(ui.get('stop').disabled, true);
  assert.deepEqual(ui.calls.map(([name]) => name), ['desktop_status']);
});

test('confirmed Start sends only confirmation and does not auto-navigate', async () => {
  const ui = await screen(state({ payload_available: true }), async (command, args, set) => {
    assert.equal(command, 'start_owned');
    assert.deepEqual(JSON.parse(JSON.stringify(args)), { confirmed: true });
    const value = state({ mode: 'owned', payload_available: true, address: 'http://127.0.0.1:23456/' }); set(value); return value;
  });
  assert.equal(ui.get('start').disabled, true);
  ui.get('confirm').checked = true;
  ui.get('confirm').handlers.change();
  assert.equal(ui.get('start').disabled, false);
  await ui.click('start');
  assert.equal(ui.get('start').disabled, true);
  assert.equal(ui.get('restart').disabled, false);
  assert.equal(ui.get('connect').disabled, true);
  assert.deepEqual(ui.navigations, []);
  await ui.click('open-owned');
  assert.deepEqual(ui.navigations, ['http://127.0.0.1:23456/']);
});

test('failed shutdown exposes uncertainty and does not enable another Start', async () => {
  const ui = await screen(state({ mode: 'owned', payload_available: true, address: 'http://127.0.0.1:23456/' }), async (command, _args, set) => {
    assert.equal(command, 'stop_owned');
    const value = state({ mode: 'failed', payload_available: true, cleanup_unknown: true, message: 'Cleanup unknown' }); set(value); return value;
  });
  await ui.click('stop');
  assert.equal(ui.get('start').disabled, true);
  assert.equal(ui.get('restart').disabled, true);
  assert.equal(ui.get('open-owned').hidden, true);
  assert.equal(ui.get('owned-message').textContent, 'Cleanup unknown');
});

test('external connection invokes only the existing fixed probe', async () => {
  const ui = await screen(state(), async command => {
    assert.equal(command, 'probe_backend');
    return { status: 'ready', version: '0.1.0', address: 'http://127.0.0.1:8000/' };
  });
  await ui.click('connect');
  assert.deepEqual(ui.navigations, ['http://127.0.0.1:8000/']);
  assert.equal(ui.calls.some(([name]) => name === 'start_owned' || name === 'stop_owned'), false);
});


test('Stop remains available during a pending owned startup, without a second Start', async () => {
  let rejectStart;
  const ui = await screen(state({ payload_available: true }), async (command, _args, set) => {
    if (command === 'start_owned') return new Promise((_resolve, reject) => { rejectStart = reject; });
    assert.equal(command, 'stop_owned');
    set(state({ payload_available: true, message: 'Stopped' }));
    rejectStart('Startup was cancelled.');
    return state({ payload_available: true, message: 'Stopped' });
  });
  ui.get('confirm').checked = true;
  ui.get('confirm').handlers.change();
  const start = ui.click('start');
  await tick();
  assert.equal(ui.get('stop').disabled, false);
  assert.equal(ui.get('start').disabled, true);
  await ui.click('stop');
  await start;
  assert.equal(ui.calls.filter(([name]) => name === 'start_owned').length, 1);
  assert.equal(ui.calls.filter(([name]) => name === 'stop_owned').length, 1);
  assert.deepEqual(ui.navigations, []);
});
