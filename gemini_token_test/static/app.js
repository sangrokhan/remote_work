let compareWireChart, compareTokenChart;

function fmtBytes(n) {
  if (n == null) return "";
  if (n < 1024) return n + " B";
  if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
  return (n / 1048576).toFixed(2) + " MB";
}

function fillTable(tbodySelector, rows, numeric = []) {
  const tb = document.querySelector(tbodySelector);
  tb.innerHTML = "";
  rows.forEach(cells => {
    const tr = document.createElement("tr");
    cells.forEach((v, i) => {
      const td = document.createElement("td");
      td.textContent = (v == null) ? "" : String(v);
      if (numeric.includes(i)) td.className = "num";
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
}

// Attach a generated file to a download link, releasing the previous blob so a
// long session doesn't leak one object URL per run.
function attachDownload(linkId, payload, filename) {
  const dl = document.getElementById(linkId);
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  if (dl.dataset.url) URL.revokeObjectURL(dl.dataset.url);
  const u = URL.createObjectURL(blob);
  dl.href = u; dl.dataset.url = u; dl.download = filename;
  dl.hidden = false;
}

const ARM_LABELS = {
  stateless: "Stateless (full resend)",
  cached: "Cached (build + reference)",
  interaction: "Interaction API (server-side state)",
  nocontext: "Stateless — no context",
  cachebuild: "Building caches",
};

const ARM_COLORS = {
  stateless: "#ff6b6b",
  cached: "#4dd4ac",
  interaction: "#5b8def",
  nocontext: "#f6c453",
};

function progressText(p) {
  if (p.stage === "pause") {
    const next = ARM_LABELS[p.next_arm] || p.next_arm;
    return `⏸ Rate-limit spacing — ${p.remaining}s / ${p.pause_total}s until ${next}…`;
  }
  const label = ARM_LABELS[p.stage] || p.stage;
  // The cached arm builds a cache per turn before it answers anything — the
  // slowest stretch of the run, so say which of the two it is in.
  const what = p.phase === "setup" ? "building cache" : "turn";
  let s = `Running… ${label}`;
  if (p.turns) s += ` — ${what} ${p.turn}/${p.turns}`;
  // Interaction turns report each SSE event, so a stall shows which stage owns it.
  if (p.event) s += ` · ${p.event} @ ${((p.at_ms || 0) / 1000).toFixed(1)}s`;
  return s;
}

// POST to a streaming endpoint and drain Server-Sent Events: forward each
// progress event to onProgress, return the final result payload.
async function runStreaming(body, onProgress, url) {
  const resp = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok || !resp.body) {
    let err = resp.status;
    try { err = (await resp.json()).error || err; } catch (e) {}
    return { __error: err };
  }
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "", result = null;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, idx).split("\n").find(l => l.startsWith("data:"));
      buf = buf.slice(idx + 2);
      if (!line) continue;
      let ev; try { ev = JSON.parse(line.slice(5).trim()); } catch (e) { continue; }
      if (ev.type === "progress") onProgress(ev);
      else if (ev.type === "done") result = ev.payload;
      else if (ev.type === "error") result = { __error: ev.error };
    }
  }
  return result;
}

// Drives a streaming run button: disables it, hides stale output, streams progress
// into the status line, and validates the final payload. Returns the result, or
// null when the run errored (the status line already says why).
async function streamedRun({ button, url, startText, hide, body, requires, missingMsg, statusId }) {
  const btn = document.getElementById(button);
  const status = document.getElementById(statusId);
  btn.disabled = true;
  status.textContent = startText;
  try {
    (hide || []).forEach(id => { document.getElementById(id).hidden = true; });
    const data = await runStreaming(body(), (p) => { status.textContent = progressText(p); }, url);
    const err = data && (data.__error || data.error);
    if (!data || err || !data[requires]) {
      status.textContent = "Error: " + (err || missingMsg);
      return null;
    }
    return data;
  } catch (e) {
    status.textContent = "Failed: " + e;
    return null;
  } finally {
    btn.disabled = false;
  }
}

// --- Interaction API capability probe ----------------------------------------
// One row per (target, model). A target that failed on auth or allowlist never
// reached the schema questions, so it gets a single explanatory row instead of a
// grid of "unsupported" that would read as a finding it isn't.
function probeRows(targets) {
  const rows = [];
  (targets || []).forEach(t => {
    const models = Object.entries(t.models || {});
    if (!models.length) {
      rows.push([t.target, "—", t.verdict, "", "", "", t.reason || t.control_message || ""]);
      return;
    }
    models.forEach(([model, e]) => rows.push([
      t.target, model,
      e.interactions_stream.verdict,
      e.interactions_nonstream.verdict,
      e.generate_content.verdict,
      e.usage_reported ? "yes" : "no",
      e.interactions_nonstream.error || e.interactions_stream.error || "",
    ]));
  });
  return rows;
}

function renderProbe(data) {
  fillTable("#probeTable tbody", probeRows(data.targets));

  const c = data.conclusion || {};
  document.getElementById("probeConclusion").textContent =
    `${data.mock ? "[MOCK] " : ""}${c.next_step}: ${c.summary}`;

  // The system-prompt answer decides how many bytes a stateful turn really costs,
  // so surface it next to the conclusion rather than burying it in the JSON.
  const sys = (data.targets || [])
    .map(t => [t.target, (t.checks || {}).system_instruction || {}])
    .find(([, s]) => s.verdict === "persisted" || s.verdict === "per_turn");
  document.getElementById("probeSystem").textContent = sys
    ? `system_instruction on ${sys[0]}: ${sys[1].verdict} — ${sys[1].meaning}`
    : "system_instruction: not probed (no model interaction succeeded).";

  attachDownload("probeJson", data, "interaction_probe.json");
  document.getElementById("probeResult").hidden = false;
}

// force=false is the page-load probe: the server serves it from cache, so a
// refresh costs nothing. The button forces a real re-run.
async function startProbe(force = true) {
  const btn = document.getElementById("startProbe");
  const status = document.getElementById("probeStatus");
  btn.disabled = true;
  status.textContent = force ? "Probing… (a handful of live calls)" : "Probing…";
  try {
    const resp = await fetch("/interaction/probe", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force }),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      status.textContent = "Error: " + (data.error || resp.status);
      return;
    }
    renderProbe(data);
    const age = data.cached ? ` (cached ${data.age_seconds}s ago)` : "";
    status.textContent = `Probe done → ${(data.conclusion || {}).next_step}${age}`;
  } catch (e) {
    status.textContent = "Failed: " + e;
  } finally {
    btn.disabled = false;
  }
  // The model list carries the probe's interaction verdicts, so it is only
  // complete once the probe has answered.
  await loadModels();
}

// --- Interaction API test (stateful, server-side history) --------------------
function renderInteraction(data) {
  const secs = (ms) => (ms == null ? "" : (ms / 1000).toFixed(1));
  fillTable("#interactionTable tbody",
    (data.records || []).map(r => [r.turn, r.question, r.response_text,
      secs(r.elapsed_ms), secs(r.first_event_ms), r.interaction_id, r.error]),
    [0, 3, 4]);
  const p = data.params || {};
  document.getElementById("interactionMeta").textContent =
    `${data.mock ? "[MOCK] " : ""}model: ${p.model || "?"} · turns: ${p.turns ?? "?"}`
    + ` · exec_id: ${data.exec_id || "?"}`;
  attachDownload("interactionChat", data.records || [],
                 `interaction_${data.exec_id || "run"}.json`);
  document.getElementById("interactionResult").hidden = false;
}

async function startInteraction() {
  const data = await streamedRun({
    button: "startInteraction",
    url: "/interaction/test",
    startText: "Interaction API…",
    statusId: "interactionStatus",
    hide: ["interactionResult", "interactionChat"],
    requires: "records",
    missingMsg: "no result",
    body: () => ({
      turns: +document.getElementById("ixTurns").value,
      model: selectedModel(),
    }),
  });
  if (!data) return;

  renderInteraction(data);
  const errs = data.records.filter(r => r.error).length;
  document.getElementById("interactionStatus").textContent =
    `${data.mock ? "[MOCK] " : ""}Interaction done. exec_id: ${data.exec_id}`
    + (errs ? ` | ${errs} turn(s) errored` : "");
  loadHistory();
}

// --- Performance comparison --------------------------------------------------
// Overlay each arm's cumulative wire bytes and input tokens on their own axes.
// x is the steady turn index; the setup cost is folded into the arm's starting
// point (metrics offsets the cumulative series), so cached does not start free.
function plotCompare(summary) {
  const arms = summary.arms || [];
  const maxLen = Math.max(0, ...arms.map(a => (summary.series[a].turns || []).length));
  const labels = Array.from({ length: maxLen }, (_, i) => i + 1);
  const mk = (key) => arms.map(a => {
    const c = ARM_COLORS[a] || "#9aa5b1";
    return {
      label: a, data: summary.series[a][key],
      borderColor: c, backgroundColor: c + "22", fill: false, tension: .2,
    };
  });
  const opts = (title, yLabel) => ({
    responsive: true,
    plugins: { title: { display: true, text: title, color: "#e6e6e6" }, legend: { labels: { color: "#cbd5e0" } } },
    scales: {
      x: { title: { display: true, text: "steady turn", color: "#8b98a5" }, ticks: { color: "#8b98a5" }, grid: { color: "#2d3748" } },
      y: { title: { display: true, text: yLabel, color: "#8b98a5" }, ticks: { color: "#8b98a5" }, grid: { color: "#2d3748" } },
    },
  });
  if (compareWireChart) compareWireChart.destroy();
  if (compareTokenChart) compareTokenChart.destroy();
  compareWireChart = new Chart(document.getElementById("compareWireChart"),
    { type: "line", data: { labels, datasets: mk("cum_wire") }, options: opts("Cumulative wire bytes (incl. setup)", "bytes") });
  compareTokenChart = new Chart(document.getElementById("compareTokenChart"),
    { type: "line", data: { labels, datasets: mk("cum_input_tokens") }, options: opts("Cumulative input tokens (incl. setup)", "tokens") });
}

// A run with a broken arm still returns numbers, and a number from a failed call
// looks like a number from a good one. Name the failing cases above the table.
function renderCompareErrors(failures, capUnavailable) {
  const el = document.getElementById("compareErrors");
  const lines = [];
  if (capUnavailable) lines.push(`⚠ capture skipped — ${capUnavailable}`);
  (failures || []).forEach(f =>
    lines.push(`✗ ${f.arm} · ${f.phase} · turn ${f.turn} — ${f.error}`));
  el.textContent = lines.join("\n");
  el.hidden = !lines.length;
}

function renderCompareArms(data) {
  const s = data.summary;
  const t = s.totals;
  const pcaps = data.pcaps || {};
  const secs = (ms) => (ms == null ? "" : (ms / 1000).toFixed(1));
  const pcapCell = (a) => {
    const c = pcaps[a];
    if (!c) return "";
    return c.download ? `${fmtBytes(c.bytes)}` : (c.error || c.note || "");
  };
  fillTable("#compareArmsTable tbody",
    (s.arms || []).map(a => [
      ARM_LABELS[a] || a,
      fmtBytes(t[a].setup_wire), fmtBytes(t[a].steady_wire), fmtBytes(t[a].total_wire),
      (t[a].total_input_tokens || 0).toLocaleString(),
      (t[a].cached_tokens || 0).toLocaleString(),
      (t[a].output_tokens || 0).toLocaleString(),
      secs(t[a].latency.mean), secs(t[a].call_ms), secs(t[a].wall_ms),
      t[a].errors, pcapCell(a),
    ]),
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

  // pcap cells carry a download link, which fillTable (text-only, by design) won't
  // render — so attach them after the fact.
  document.querySelectorAll("#compareArmsTable tbody tr").forEach((tr, i) => {
    const arm = (s.arms || [])[i];
    const c = pcaps[arm];
    if (!c || !c.download) return;
    const td = tr.cells[11];
    td.innerHTML = "";
    const a = document.createElement("a");
    a.href = c.download; a.className = "pcap-link"; a.download = "";
    a.textContent = `⬇ ${fmtBytes(c.bytes)}`;
    td.appendChild(a);
  });

  renderCompareErrors(s.failures, data.capture_unavailable);
  plotCompare(s);

  // Raw request/response bodies live server-side: a run of ten turns across three
  // arms is megabytes of JSON, too much to hold in a blob just to hand back.
  if (data.exec_id) {
    const id = encodeURIComponent(data.exec_id);
    const json = document.getElementById("compareArmsJson");
    json.href = `/download/comparison/${id}.json`;
    json.hidden = false;
    const csv = document.getElementById("compareArmsCsv");
    csv.href = `/download/comparison/${id}.csv`;
    csv.hidden = false;
  }
  document.getElementById("compareArmsResult").hidden = false;
}

async function startCompare() {
  const arms = Array.from(document.querySelectorAll(".cmpArm:checked")).map(c => c.value);
  const data = await streamedRun({
    button: "startCompare",
    url: "/compare/stream",
    startText: "Comparison…",
    statusId: "compareArmsStatus",
    hide: ["compareArmsResult", "compareArmsJson", "compareArmsCsv"],
    requires: "summary",
    missingMsg: "no result",
    body: () => ({
      turns: +document.getElementById("cmpTurns").value,
      pause_seconds: +document.getElementById("cmpPause").value,
      capture: document.getElementById("cmpCapture").checked,
      model: selectedModel(),
      arms: arms.length ? arms : undefined,
    }),
  });
  if (!data) return;

  renderCompareArms(data);
  const errs = (data.summary.failures || []).length;
  document.getElementById("compareArmsStatus").textContent =
    `${data.mock ? "[MOCK] " : ""}Comparison done. exec_id: ${data.exec_id}`
    + (errs ? ` | ${errs} call(s) errored — see the list above` : "");
  loadHistory();
}

// --- Execution history viewer ------------------------------------------------
async function loadHistory() {
  const resp = await fetch("/history");
  const data = await resp.json();
  const runs = data.runs || [];
  document.getElementById("histSource").textContent =
    `source: ${data.source}` + (data.dummy ? " — showing DUMMY data (no history found)" : "");

  // A comparison run has no single headline number, so show the stateless arm as
  // the reference and let "view" open the full per-arm breakdown.
  const ref = (t) => (t && t.stateless) || {};
  const tb = document.querySelector("#histTable tbody");
  tb.innerHTML = "";
  for (const r of runs) {
    const flags = (r.dummy ? "DUMMY " : "") + (r.mock ? "MOCK" : "");
    const t = ref(r.totals);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.exec_id}</td><td>${r.mode || ""}</td><td>${r.timestamp || ""}</td>
      <td>${t.total_wire != null ? fmtBytes(t.total_wire) : (r.totals?.wire_bytes != null ? fmtBytes(r.totals.wire_bytes) : "")}</td>
      <td>${t.total_input_tokens != null ? t.total_input_tokens.toLocaleString() : ""}</td>
      <td>${flags}</td><td><button data-id="${r.exec_id}" class="viewBtn">view</button>
      <button data-id="${r.exec_id}" class="delBtn"${r.dummy ? " disabled title='dummy row'" : ""}>🗑 delete</button></td>`;
    tb.appendChild(tr);
  }
  document.querySelectorAll(".viewBtn").forEach(b =>
    b.addEventListener("click", () => viewExec(b.dataset.id)));
  document.querySelectorAll(".delBtn").forEach(b =>
    b.addEventListener("click", () => deleteExec(b.dataset.id)));
}

async function deleteExec(execId) {
  if (!confirm("Delete execution " + execId + "?\nThis permanently removes its stored record.")) return;
  const resp = await fetch("/history/" + encodeURIComponent(execId), { method: "DELETE" });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || !data.ok) { alert("Delete failed: " + (data.error || resp.status)); return; }
  // Clear the detail pane if it was showing the run we just deleted.
  const meta = document.getElementById("histDetailMeta");
  if (meta.textContent.startsWith(execId)) {
    meta.textContent = "";
    document.getElementById("histDetail").hidden = true;
    document.getElementById("histDownload").hidden = true;
  }
  loadHistory();
}

async function viewExec(execId) {
  const resp = await fetch("/history/" + encodeURIComponent(execId));
  if (!resp.ok) return;
  const doc = await resp.json();
  document.getElementById("histDetailMeta").textContent =
    `${doc.exec_id} · ${doc.mode} · ${doc.timestamp}`
    + (doc.dummy ? " · DUMMY" : "") + (doc.mock ? " · MOCK" : "");
  const pre = document.getElementById("histDetail");
  pre.hidden = false;
  pre.textContent = JSON.stringify(doc, null, 2);
  const dl = document.getElementById("histDownload");
  dl.href = "/download/run/" + encodeURIComponent(execId);
  dl.hidden = false;
  // Replot a stored comparison so an old run can be re-read without re-running it.
  const sm = doc.summary || {};
  if (sm.mode === "comparison" && sm.series) {
    renderCompareArms({ summary: sm, exec_id: doc.exec_id, mock: doc.mock });
  }
}

// --- Endpoint inspector ------------------------------------------------------
async function inspect() {
  const btn = document.getElementById("inspect");
  const status = document.getElementById("iStatus");
  const out = document.getElementById("iResult");
  const dl = document.getElementById("iDownload");
  dl.hidden = true;
  const url = document.getElementById("iUrl").value.trim();
  if (!url) { status.textContent = "Enter a URL."; return; }
  btn.disabled = true;
  status.textContent = "Inspecting…";
  try {
    const body = {
      method: document.getElementById("iMethod").value,
      url,
      headers: document.getElementById("iHeaders").value,
      body: document.getElementById("iBody").value,
      include_bodies: document.getElementById("iBodies").checked,
      allow_private: document.getElementById("iPrivate").checked,
    };
    const resp = await fetch("/inspect", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    out.hidden = false;
    out.textContent = JSON.stringify(data, null, 2);
    if (!data.ok) { status.textContent = "✗ " + (data.error || resp.status); return; }
    const hints = data.protocol_hints && data.protocol_hints.length
      ? " | protocol: " + data.protocol_hints.join(", ") : "";
    status.textContent = `✓ ${data.status} · ${data.elapsed_ms}ms · `
      + `wire ${fmtBytes(data.wire_sent)}↑ ${fmtBytes(data.wire_recv)}↓${hints}`;
    if (data.download) { dl.href = data.download; dl.hidden = false; }
  } catch (e) {
    status.textContent = "Failed: " + e;
  } finally {
    btn.disabled = false;
  }
}

// --- Model dropdown + search -------------------------------------------------
const CUSTOM = "__custom__";
let allModels = [];

function selectedModel() {
  const sel = document.getElementById("model");
  if (sel.value === CUSTOM) return document.getElementById("modelCustom").value.trim();
  return sel.value;
}

function renderModels(filter) {
  const sel = document.getElementById("model");
  const prev = sel.value;
  const f = (filter || "").toLowerCase();
  const shown = allModels.filter(m => !f || m.id.toLowerCase().includes(f) || (m.label || "").toLowerCase().includes(f));
  sel.innerHTML = "";
  for (const m of shown) {
    const o = document.createElement("option");
    o.value = m.id; o.textContent = m.label || m.id;
    sel.appendChild(o);
  }
  const custom = document.createElement("option");
  custom.value = CUSTOM; custom.textContent = "custom…";
  sel.appendChild(custom);
  if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
  toggleCustom();
}

function toggleCustom() {
  const sel = document.getElementById("model");
  document.getElementById("modelCustom").hidden = sel.value !== CUSTOM;
}

async function loadModels() {
  try {
    const resp = await fetch("/models");
    const data = await resp.json();
    allModels = data.models || [];
    renderModels("");
    const sel = document.getElementById("model");
    if (data.default && [...sel.options].some(o => o.value === data.default)) sel.value = data.default;
    toggleCustom();
  } catch (e) { /* keep server-rendered default */ }
}

document.getElementById("model").addEventListener("change", toggleCustom);
document.getElementById("modelFilter").addEventListener("input", e => renderModels(e.target.value));
document.getElementById("modelRefresh").addEventListener("click", loadModels);
document.getElementById("startProbe").addEventListener("click", () => startProbe(true));
document.getElementById("startInteraction").addEventListener("click", startInteraction);
document.getElementById("startCompare").addEventListener("click", startCompare);
document.getElementById("refresh").addEventListener("click", loadHistory);
document.getElementById("inspect").addEventListener("click", inspect);

// Probe on load (served from the server's cache on a refresh), then fill the model
// list from it — startProbe reloads the models once the verdicts are in.
loadModels();
startProbe(false);
loadHistory();
