import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { generatePayload, createServer } from './server.js';

let server;
let port;

// Start a test server on a random port before all tests.
before(async () => {
  server = createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  port = server.address().port;
});

// Shut down the test server after all tests.
after(async () => {
  await new Promise(resolve => server.close(resolve));
});

test('generatePayload returns exactly 300 chars', () => {
  // base64(225 bytes) = 225 * 4/3 = 300 chars, no padding
  assert.equal(generatePayload().length, 300);
});

test('generatePayload output is valid base64', () => {
  // base64(225 bytes) = exactly 300 chars, no padding (225 % 3 === 0)
  assert.match(generatePayload(), /^[A-Za-z0-9+/]+$/);
});

test('GET / returns 200 with HTML containing EventSource', async () => {
  const res = await fetch(`http://127.0.0.1:${port}/`);
  assert.equal(res.status, 200);
  const body = await res.text();
  assert.ok(body.includes('EventSource'), 'must include EventSource JS');
  assert.ok(body.includes('/stream'), 'must reference /stream endpoint');
});

test('GET /stream sets text/event-stream content-type', async () => {
  const ac = new AbortController();
  const res = await fetch(`http://127.0.0.1:${port}/stream`, {
    signal: ac.signal,
  });
  // Headers are fully available once fetch() resolves; abort stops body streaming.
  assert.equal(res.headers.get('content-type'), 'text/event-stream');
  ac.abort();
});
