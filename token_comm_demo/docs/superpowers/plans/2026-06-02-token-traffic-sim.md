# Token Traffic Simulation Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Go HTTP server that streams SSE events (~300 bytes each, ~1000/sec) so a phone browser can receive token-like traffic and DroidPCap can capture it.

**Architecture:** Single `main.go` file with two HTTP handlers: `/` serves an inline HTML page with a JS `EventSource` client, `/stream` sends SSE events at 1ms intervals using `time.Ticker` and `http.Flusher`. A `generatePayload()` helper is extracted for unit testing.

**Tech Stack:** Go 1.21+ stdlib only (`net/http`, `encoding/base64`, `crypto/rand`, `time`)

---

## File Map

| File | Role |
|------|------|
| `go.mod` | Module declaration |
| `main.go` | HTTP server, both handlers, payload generator |
| `main_test.go` | Unit tests for payload size and `/` response |

---

### Task 1: Initialize Go module

**Files:**
- Create: `go.mod`

- [ ] **Step 1: Run go mod init**

```bash
cd /home/han/.openclaw/workspace/remote_work/token_comm_demo
go mod init token_comm_demo
```

Expected output:
```
go: creating new go.mod: module token_comm_demo
```

- [ ] **Step 2: Verify go.mod**

```bash
cat go.mod
```

Expected:
```
module token_comm_demo

go 1.21
```

- [ ] **Step 3: Commit**

```bash
git add go.mod
git commit -m "chore(token_comm_demo): init go module"
```

---

### Task 2: Write failing tests

**Files:**
- Create: `main_test.go`

- [ ] **Step 1: Write tests**

Create `main_test.go`:

```go
package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestGeneratePayload verifies payload is exactly 300 chars (base64 of 225 bytes).
func TestGeneratePayload(t *testing.T) {
	p := generatePayload()
	if len(p) != 300 {
		t.Errorf("expected 300 chars, got %d", len(p))
	}
}

// TestGeneratePayloadIsBase64 verifies payload contains only valid base64 characters.
func TestGeneratePayloadIsBase64(t *testing.T) {
	p := generatePayload()
	for _, c := range p {
		if !strings.ContainsRune("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=", c) {
			t.Errorf("non-base64 char in payload: %q", c)
		}
	}
}

// TestRootHandler verifies / returns 200 and HTML with EventSource.
func TestRootHandler(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	rootHandler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, "EventSource") {
		t.Error("HTML page must include EventSource JS")
	}
	if !strings.Contains(body, "/stream") {
		t.Error("HTML page must reference /stream endpoint")
	}
}

// TestStreamHeaders verifies /stream sets correct SSE headers.
// Context is cancelled before the handler starts so it exits immediately after
// writing headers, without looping on the ticker.
func TestStreamHeaders(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled — handler exits as soon as it checks Done()

	req := httptest.NewRequest(http.MethodGet, "/stream", nil).WithContext(ctx)
	w := &flusherRecorder{ResponseRecorder: httptest.NewRecorder()}

	done := make(chan struct{})
	go func() {
		streamHandler(w, req)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("handler did not exit after context cancellation")
	}

	ct := w.Header().Get("Content-Type")
	if ct != "text/event-stream" {
		t.Errorf("expected text/event-stream, got %q", ct)
	}
}

// flusherRecorder wraps httptest.ResponseRecorder to implement http.Flusher.
type flusherRecorder struct {
	*httptest.ResponseRecorder
	flushed bool
}

func (f *flusherRecorder) Flush() {
	f.flushed = true
}
```

- [ ] **Step 2: Run tests — expect compile failure (main.go missing)**

```bash
cd /home/han/.openclaw/workspace/remote_work/token_comm_demo
go test ./... 2>&1 | head -20
```

Expected: build error — `generatePayload`, `rootHandler`, `streamHandler` undefined.

- [ ] **Step 3: Commit test file**

```bash
git add main_test.go
git commit -m "test(token_comm_demo): add unit tests for payload, handlers"
```

---

### Task 3: Implement main.go

**Files:**
- Create: `main.go`

- [ ] **Step 1: Write main.go**

```go
package main

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"log"
	"net/http"
	"time"
)

// generatePayload creates a 300-char base64 string from 225 random bytes.
// base64(225 bytes) = 300 chars exactly (225 * 4/3 = 300, no padding needed).
func generatePayload() string {
	raw := make([]byte, 225)
	rand.Read(raw) // crypto/rand; ignoring error — never fails on Linux
	return base64.StdEncoding.EncodeToString(raw)
}

// rootHandler serves a simple HTML page with an EventSource client.
// The page connects to /stream and displays a live event counter and rate.
func rootHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprint(w, `<!DOCTYPE html>
<html>
<head><title>Token Traffic Sim</title></head>
<body>
<h2>Token Traffic Simulator</h2>
<p>Events received: <span id="count">0</span></p>
<p>Rate: <span id="rate">0</span> events/sec</p>
<script>
  var count = 0, lastCount = 0;
  var es = new EventSource('/stream');
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
</html>`)
}

// streamHandler sends SSE events at ~1000 events/sec.
// Each event payload is ~300 bytes (300-char base64 string).
// Flush() is called after every write to prevent buffering from delaying delivery.
func streamHandler(w http.ResponseWriter, r *http.Request) {
	// SSE requires these headers; CORS header allows phone browser access.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	// Ticker fires every 1ms → ~1000 events/sec.
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// SSE event format: "data: <payload>\n\n"
			fmt.Fprintf(w, "data: %s\n\n", generatePayload())
			// Flush immediately so the TCP stack sends without waiting for more data.
			flusher.Flush()
		case <-r.Context().Done():
			// Client disconnected or request cancelled.
			return
		}
	}
}

func main() {
	http.HandleFunc("/", rootHandler)
	http.HandleFunc("/stream", streamHandler)

	addr := ":8080"
	log.Printf("listening on %s — open http://<your-ip>:8080/ on your phone", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
```

- [ ] **Step 2: Run tests — expect PASS**

```bash
cd /home/han/.openclaw/workspace/remote_work/token_comm_demo
go test ./... -v
```

Expected:
```
=== RUN   TestGeneratePayload
--- PASS: TestGeneratePayload (0.00s)
=== RUN   TestGeneratePayloadIsBase64
--- PASS: TestGeneratePayloadIsBase64 (0.00s)
=== RUN   TestRootHandler
--- PASS: TestRootHandler (0.00s)
=== RUN   TestStreamHeaders
--- PASS: TestStreamHeaders (0.00s)
PASS
ok  	token_comm_demo	...
```

- [ ] **Step 3: Build binary to confirm no compile errors**

```bash
go build -o token_sim .
ls -lh token_sim
```

Expected: binary present, ~4–6 MB.

- [ ] **Step 4: Commit**

```bash
git add main.go
git commit -m "feat(token_comm_demo): add SSE server, 1000 pps, 300-byte events"
```

---

### Task 4: Run and verify

**Files:** none — runtime verification only.

- [ ] **Step 1: Get host IP (phone must reach this)**

```bash
ip -4 addr show | grep inet | grep -v 127 | awk '{print $2}'
```

Note the IP — you'll open `http://<IP>:8080/` on your phone.

- [ ] **Step 2: Run server**

```bash
./token_sim
```

Expected log line:
```
listening on :8080 — open http://<your-ip>:8080/ on your phone
```

- [ ] **Step 3: Open on phone browser**

Navigate to `http://<IP>:8080/`. Verify:
- Page loads with "Token Traffic Simulator"
- Counter increments rapidly
- Rate display shows ~1000 events/sec

- [ ] **Step 4: Start DroidPCap capture on phone**

Filter: `tcp port 8080`  
Verify captured packets show ~300+ byte payloads at ~1ms intervals.

- [ ] **Step 5: Stop server and clean binary**

```bash
# Ctrl+C to stop server, then:
rm token_sim
git add -u
git commit -m "chore(token_comm_demo): remove built binary"
```
