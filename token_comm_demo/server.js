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
  createServer().listen(10000, () => {
    console.log('listening on :10000 — open http://<your-ip>:10000/ on your phone');
  });
}
