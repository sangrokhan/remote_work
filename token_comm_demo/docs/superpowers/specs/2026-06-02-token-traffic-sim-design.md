# Token Traffic Simulation Server — Design Spec

**Date:** 2026-06-02  
**Project:** `token_comm_demo`

## Goal

Simulate LLM token streaming traffic so TCP behavior (congestion, out-of-order delivery, retransmission) can be studied via packet capture on a phone (DroidPCap).

## Architecture

Single Go binary. No external dependencies (stdlib only).

```
Phone browser
    │  HTTP GET /          → HTML page with EventSource JS
    │  HTTP GET /stream    → SSE stream (1000 events/sec, ~300 bytes/event)
    ▼
Go HTTP server (this Linux host)
```

## Endpoints

| Path | Description |
|------|-------------|
| `GET /` | Serves inline HTML; JS opens `EventSource("/stream")` and displays live counter |
| `GET /stream` | SSE endpoint; streams events at 1ms intervals |

## Packet Design

- Raw payload: 225 random bytes per event
- Encoded: base64(225 bytes) = 300 chars — satisfies ~300-byte SSE `data:` line
- Random bytes regenerated each event → no compression benefit, realistic entropy
- SSE frame: `data: <300chars>\n\n` = ~308 bytes on wire before TCP/IP headers

## Timing

- `time.NewTicker(time.Millisecond)` — fires ~1000 times/sec
- `http.Flusher.Flush()` called after every write — bypasses Go's response buffer
- OS scheduler jitter expected: ±0.1–0.5ms on Linux; acceptable for TCP study

## Files

```
token_comm_demo/
└── main.go      # entire implementation, single file
```

## Success Criteria

1. Phone browser connects to `http://<host-ip>:8080/` without error
2. Event counter increments visibly (~1000/sec)
3. DroidPCap captures packets of ~300+ bytes at ~1ms intervals on port 8080
4. No server crash under continuous single-client load
