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
    // Disable Nagle's algorithm so each res.write() sends immediately
    // without waiting to coalesce with subsequent writes.
    req.socket.setNoDelay(true);
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
<p>
  <button id="btn-start" onclick="startStream()">Start</button>
  <button id="btn-stop" onclick="stopStream()" disabled>Stop</button>
</p>
<script>
  var count = 0, lastCount = 0, es = null;

  // Avoid naming these start/stop — those are reserved browser globals.
  function startStream() {
    if (es) return;
    es = new EventSource('/stream');
    // Count only — no DOM write per event (too slow at high pps).
    es.onmessage = function() { count++; };
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
  }

  function stopStream() {
    if (!es) return;
    es.close();
    es = null;
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
  }

  // Update both counter and rate once per second — decoupled from message rate.
  setInterval(function() {
    var rate = count - lastCount;
    document.getElementById('count').textContent = count;
    document.getElementById('rate').textContent = rate;
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

  // setImmediate loop: one write per event loop tick, as fast as possible.
  // Each tick is its own I/O phase, so the socket flushes between writes —
  // no timer batching, no Nagle coalescing.
  let active = true;
  function send() {
    if (!active || req.destroyed) return;
    res.write(`data: ${generatePayload()}\n\n`);
    setImmediate(send);
  }
  setImmediate(send);

  req.on('close', () => { active = false; });
}

// Start the server only when this file is run directly (not when imported by tests).
if (import.meta.url === `file://${process.argv[1]}`) {
  createServer().listen(10000, () => {
    console.log('listening on :10000 — open http://<your-ip>:10000/ on your phone');
  });
}
