# Token Traffic Simulation Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Node.js HTTP server that streams SSE events (~300 bytes each, ~1000/sec) so a phone browser can receive token-like traffic and DroidPCap can capture it.

**Architecture:** `server.js` exports `generatePayload()` and `createServer()` so they can be unit-tested. The server starts only when run directly. `server.test.js` uses Node's built-in `node:test` runner and `fetch` (both available in Node 22, no external deps).

**Tech Stack:** Node.js 22, stdlib only (`node:http`, `node:crypto`, `node:test`)

---

## File Map

| File | Role |
|------|------|
| `package.json` | Module type (ESM), test + start scripts |
| `server.js` | HTTP server, SSE stream, payload generator |
| `server.test.js` | Unit + integration tests |

---

### Task 1: Initialize package.json

**Files:**
- Create: `token_comm_demo/package.json`

- [ ] **Step 1: Write package.json**

Working dir: `/home/han/.openclaw/workspace/remote_work/.claude/worktrees/token-traffic-sim/token_comm_demo`

Create `package.json`:

```json
{
  "name": "token_comm_demo",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "test": "node --test server.test.js",
    "start": "node server.js"
  }
}
```

- [ ] **Step 2: Commit**

Run from worktree root `/home/han/.openclaw/workspace/remote_work/.claude/worktrees/token-traffic-sim`:

```bash
git add token_comm_demo/package.json
git commit -m "chore(token_comm_demo): init package.json"
```

---

### Task 2: Write failing tests

**Files:**
- Create: `token_comm_demo/server.test.js`

- [ ] **Step 1: Write server.test.js**

Working dir: `/home/han/.openclaw/workspace/remote_work/.claude/worktrees/token-traffic-sim/token_comm_demo`

Create `server.test.js`:

```javascript
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
  assert.match(generatePayload(), /^[A-Za-z0-9+/=]+$/);
});

test('GET / returns 200 with HTML containing EventSource', async () => {
  const res = await fetch(`http://127.0.0.1:${port}/`);
  assert.equal(res.status, 200);
  const body = await res.text();
  assert.ok(body.includes('EventSource'), 'must include EventSource JS');
  assert.ok(body.includes('/stream'), 'must reference /stream endpoint');
});

test('GET /stream sets text/event-stream content-type', async () => {
  // Abort immediately after headers arrive — we only need to check headers.
  const ac = new AbortController();
  try {
    const res = await fetch(`http://127.0.0.1:${port}/stream`, {
      signal: ac.signal,
    });
    assert.equal(res.headers.get('content-type'), 'text/event-stream');
  } finally {
    ac.abort();
  }
});
```

- [ ] **Step 2: Run tests — expect failure (server.js missing)**

```bash
cd /home/han/.openclaw/workspace/remote_work/.claude/worktrees/token-traffic-sim/token_comm_demo
npm test 2>&1 | head -20
```

Expected: error — `Cannot find module './server.js'`

- [ ] **Step 3: Commit**

```bash
git add token_comm_demo/server.test.js
git commit -m "test(token_comm_demo): add unit tests for payload and HTTP handlers"
```

---

### Task 3: Implement server.js

**Files:**
- Create: `token_comm_demo/server.js`

- [ ] **Step 1: Write server.js**

Working dir: `/home/han/.openclaw/workspace/remote_work/.claude/worktrees/token-traffic-sim/token_comm_demo`

Create `server.js`:

```javascript
import http from 'node:http';
import crypto from 'node:crypto';

// generatePayload: 225 random bytes encoded as base64 = exactly 300 chars.
// Random bytes ensure no HTTP/TCP compression reduces packet size.
export function generatePayload() {
  return crypto.randomBytes(225).toString('base64');
}

// createServer: returns an http.Server with / and /stream routes.
// Exported so tests can bind to a random port without side effects.
export function createServer() {
  return http.createServer((req, res) => {
    if (req.url === '/stream') {
      streamHandler(req, res);
    } else {
      rootHandler(res);
    }
  });
}

// rootHandler: HTML page with EventSource client that connects to /stream
// and displays a live event counter and per-second rate.
function rootHandler(res) {
  res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  res.end(`<!DOCTYPE html>
<html>
<head><title>Token Traffic Sim</title></head>
<body>
<h2>Token Traffic Simulator</h2>
<p>Events received: <span id="count">0</span></p>
<p>Rate: <span id="rate">0</span> events/sec</p>
<script>
  var count = 0, lastCount = 0;
  var es = new EventSource('/stream');
  // Increment counter on every SSE message.
  es.onmessage = function() {
    count++;
    document.getElementById('count').textContent = count;
  };
  // Update rate display every second.
  setInterval(function() {
    document.getElementById('rate').textContent = count - lastCount;
    lastCount = count;
  }, 1000);
</script>
</body>
</html>`);
}

// streamHandler: sends SSE events at ~1000 events/sec (~300 bytes each).
// Node.js writable streams flush each res.write() immediately on keep-alive
// connections, so no explicit flush is needed.
function streamHandler(req, res) {
  // SSE required headers; CORS header lets any origin (phone browser) connect.
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*',
  });

  // Fire every 1ms → ~1000 events/sec.
  // Each SSE frame: "data: <300 chars>\n\n" ≈ 308 bytes on wire.
  const timer = setInterval(() => {
    res.write(`data: ${generatePayload()}\n\n`);
  }, 1);

  // Clear the interval when the client disconnects to prevent memory leaks.
  req.on('close', () => clearInterval(timer));
}

// Start the server only when this file is run directly (not when imported by tests).
if (import.meta.url === `file://${process.argv[1]}`) {
  createServer().listen(8080, () => {
    console.log('listening on :8080 — open http://<your-ip>:8080/ on your phone');
  });
}
```

- [ ] **Step 2: Run tests — expect all 4 PASS**

```bash
cd /home/han/.openclaw/workspace/remote_work/.claude/worktrees/token-traffic-sim/token_comm_demo
npm test
```

Expected output:
```
▶ generatePayload returns exactly 300 chars
  ✔ generatePayload returns exactly 300 chars (Xms)
▶ generatePayload output is valid base64
  ✔ generatePayload output is valid base64 (Xms)
▶ GET / returns 200 with HTML containing EventSource
  ✔ GET / returns 200 with HTML containing EventSource (Xms)
▶ GET /stream sets text/event-stream content-type
  ✔ GET /stream sets text/event-stream content-type (Xms)
ℹ tests 4
ℹ pass 4
ℹ fail 0
```

- [ ] **Step 3: Commit**

```bash
git add token_comm_demo/server.js
git commit -m "feat(token_comm_demo): add SSE server, 1000 pps, 300-byte events"
```

---

### Task 4: Run and verify

**Files:** none — runtime verification only.

- [ ] **Step 1: Get host IP**

```bash
ip -4 addr show | grep inet | grep -v 127 | awk '{print $2}' | cut -d/ -f1
```

Note the IP — open `http://<IP>:8080/` on phone.

- [ ] **Step 2: Start server**

```bash
cd /home/han/.openclaw/workspace/remote_work/.claude/worktrees/token-traffic-sim/token_comm_demo
node server.js
```

Expected:
```
listening on :8080 — open http://<your-ip>:8080/ on your phone
```

- [ ] **Step 3: Verify on phone**

Open `http://<IP>:8080/` in phone browser. Confirm:
- Page loads "Token Traffic Simulator"
- Counter increments rapidly
- Rate shows ~1000 events/sec

- [ ] **Step 4: Capture with DroidPCap**

Filter: `tcp port 8080`
Verify packets ~300+ bytes at ~1ms intervals.
