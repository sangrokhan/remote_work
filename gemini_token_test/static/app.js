let tokenChart, byteChart;

function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
  return (n / 1048576).toFixed(2) + " MB";
}

function modeColor(mode) {
  if (mode === "stateless") return "#ff6b6b";
  if (mode === "stateful") return "#4dd4ac";
  return "#5b8def";
}

// items: [{label, series, color}]. Plots cum_tokens + cum_wire_bytes.
function plot(items) {
  const maxLen = Math.max(0, ...items.map(i => (i.series.turns || []).length));
  const labels = Array.from({ length: maxLen }, (_, i) => i + 1);
  const mk = (key) => items.map(i => ({
    label: i.label, data: i.series[key],
    borderColor: i.color, backgroundColor: i.color + "22",
    fill: items.length === 1, tension: .2,
  }));
  const opts = (title, yLabel) => ({
    responsive: true,
    plugins: { title: { display: true, text: title, color: "#e6e6e6" }, legend: { labels: { color: "#cbd5e0" } } },
    scales: {
      x: { title: { display: true, text: "turn", color: "#8b98a5" }, ticks: { color: "#8b98a5" }, grid: { color: "#2d3748" } },
      y: { title: { display: true, text: yLabel, color: "#8b98a5" }, ticks: { color: "#8b98a5" }, grid: { color: "#2d3748" } },
    },
  });
  if (tokenChart) tokenChart.destroy();
  if (byteChart) byteChart.destroy();
  tokenChart = new Chart(document.getElementById("tokenChart"),
    { type: "line", data: { labels, datasets: mk("cum_tokens") }, options: opts("Cumulative tokens", "tokens") });
  byteChart = new Chart(document.getElementById("byteChart"),
    { type: "line", data: { labels, datasets: mk("cum_wire_bytes") }, options: opts("Cumulative wire bytes", "bytes") });
}

function renderSummary(totals, mock, dummy) {
  const el = document.getElementById("summary");
  el.hidden = false;
  const badges = (dummy ? `<p class="badge warn">⚠ DUMMY DATA — placeholder, not a real run</p>` : "")
    + (mock ? `<p class="badge mock">⚠ MOCK RESULT — synthetic data, no real traffic/cost</p>` : "");
  el.innerHTML = `
    <h2>Result — <span class="ok">${totals.mode}</span></h2>
    ${badges}
    <p><span class="big">${(totals.tokens || 0).toLocaleString()}</span> tokens ·
       wire ${fmtBytes(totals.wire_bytes || 0)}</p>
    <p><strong>Cost estimate</strong> (@ ${totals.price_per_token}/tok): $${totals.cost_usd}</p>
    <p class="sub">Compare modes by running the other mode, then pick both in the history "Compare" box below.</p>
  `;
}

function renderSummary3(t, mock, pcaps) {
  const el = document.getElementById("summary");
  el.hidden = false;
  const badge = mock ? `<p class="badge mock">⚠ MOCK RESULT — synthetic data</p>` : "";
  let pcapHtml = "";
  const ok = pcaps && Object.entries(pcaps).filter(([k, v]) => v && v.download);
  if (ok && ok.length) {
    pcapHtml = "<p><strong>pcaps (per stage):</strong> " + ok.map(([k, v]) =>
      `<a class="pcap-link" href="${v.download}" download>⬇ ${k} (${fmtBytes(v.bytes)})</a>`).join(" ") + "</p>";
  } else if (pcaps && Object.keys(pcaps).length) {
    pcapHtml = `<p class="sub">capture ran but produced no packets (mock has no real traffic, or no NET_RAW).</p>`;
  }
  el.innerHTML = `
    <h2>Result — 3-stage caching (stateless → cache → stateful)</h2>
    ${badge}
    <p><strong>Traffic (wire bytes):</strong> stateless ${fmtBytes(t.stateless_wire)}
       vs stateful <span class="ok">${fmtBytes(t.stateful_wire)}</span>
       → <span class="big">${t.wire_ratio}×</span> less sent</p>
    <p><strong>Content length (payload):</strong> stateless ${fmtBytes(t.stateless_content)}
       vs stateful ${fmtBytes(t.stateful_content)} → ${t.content_ratio}×</p>
    <p class="sub">caches used: ${t.caches_used} · cached tokens: ${(t.cached_tokens || 0).toLocaleString()}</p>
    <p class="sub">stateful sends only the new question; the prefix is server-side in the cache.</p>
    ${pcapHtml}
  `;
}

// Render tcpdump capture stats (captured / received / dropped) per stage. `entries`
// is a {label: captureResult} map — one entry for single mode, one per stage for 3-stage.
function renderCaptureLog(entries) {
  const box = document.getElementById("captureLog");
  const rows = [];
  Object.entries(entries || {}).forEach(([label, v]) => {
    if (!v || !v.stats || !Object.keys(v.stats).length) return;
    const d = v.dropped || 0;
    const cls = d > 0 ? "caplog warn" : "caplog";
    rows.push(`<div class="${cls}">${label}: `
      + `${v.stats.captured ?? "?"} captured · `
      + `${v.stats.received_by_filter ?? "?"} recv by filter · `
      + `<strong>${d} dropped</strong> · snaplen ${v.snaplen ?? "?"}`
      + (d > 0 ? ` — capture loss (expect “ACKed unseen segment” warnings)` : "")
      + `</div>`);
  });
  if (!rows.length) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  box.innerHTML = `<strong>Capture log (tcpdump)</strong>${rows.join("")}`;
}

function renderDetail(series, mode) {
  const tb = document.querySelector("#detail tbody");
  tb.innerHTML = "";
  (series.turns || []).forEach((turn, i) => {
    const prompt = series.per_turn_prompt_tokens[i];
    const total = series.per_turn_tokens[i];
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${mode}</td><td>${turn}</td><td>${prompt}</td>
      <td>${total - prompt}</td><td>${total}</td><td>${series.per_turn_wire_bytes[i]}</td><td></td>`;
    tb.appendChild(tr);
  });
}

// 3-stage side-by-side table: one row per step, columns query / stateless resp /
// stateful resp. Uses textContent so response text can't break the layout or inject HTML.
function renderCompare3(rows, execId) {
  const sec = document.getElementById("compare3");
  const tb = document.querySelector("#compareTable tbody");
  tb.innerHTML = "";
  (rows || []).forEach(r => {
    const tr = document.createElement("tr");
    [r.turn, r.query, r.stateless_response, r.nocontext_response, r.stateful_response].forEach((v, i) => {
      const td = document.createElement("td");
      td.textContent = (v == null) ? "" : String(v);
      if (i === 0) td.className = "num";
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  const csv = document.getElementById("compareCsv");
  const has = !!(rows && rows.length);
  if (has && execId) {
    csv.href = "/download/compare/" + encodeURIComponent(execId);
    csv.hidden = false;
  } else {
    csv.hidden = true;
  }
  sec.hidden = !has;
}

const STAGE_LABELS = {
  stateless: "Stateless (full resend)",
  nocontext: "Stateless — no context",
  cachebuild: "Building caches",
  stateful: "Stateful (cache + question)",
};

function progressText(p) {
  if (p.stage === "pause") return `⏸ Pausing between stages — ${p.turn}s left (rate-limit spacing)…`;
  const label = STAGE_LABELS[p.stage] || p.stage;
  return `Running… ${label} — turn ${p.turn}/${p.turns}`;
}

// POST to the streaming endpoint and drain Server-Sent Events: forward each
// per-turn progress event to onProgress, return the final result payload.
async function runStreaming(body, onProgress) {
  const resp = await fetch("/run/stream", {
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

async function start() {
  const btn = document.getElementById("start");
  const status = document.getElementById("status");
  btn.disabled = true;
  status.textContent = "Running…";
  try {
    document.getElementById("pcapLink").hidden = true;
    document.getElementById("chatLink").hidden = true;
    document.getElementById("captureLog").hidden = true;
    document.getElementById("compare3").hidden = true;
    const body = {
      mode: document.getElementById("mode").value,
      turns: +document.getElementById("turns").value,
      model: selectedModel(),
      capture: document.getElementById("capture").checked,
      pause_seconds: +document.getElementById("pauseSeconds").value,
    };
    const data = await runStreaming(body, (p) => { status.textContent = progressText(p); });
    if (!data || data.__error || data.error || !data.summary) {
      status.textContent = "Error: " + ((data && (data.__error || data.error))
        || "stream ended early (server timeout or Vertex rate limit?)");
      return;
    }

    const s = data.summary;
    if (s.mode === "caching-3stage") {
      renderSummary3(s.totals, data.mock, data.pcaps);
      renderCaptureLog(data.pcaps);
      plot([
        { label: "stateless (full resend)", series: s.stateless_series, color: "#ff6b6b" },
        { label: "stateful (cache + question)", series: s.stateful_series, color: "#4dd4ac" },
      ]);
      renderCompare3(data.comparison, data.exec_id);
      document.querySelector("#detail tbody").innerHTML = "";
    } else {
      renderSummary(s.totals, data.mock, false);
      renderCaptureLog(data.capture ? { [s.mode]: data.capture } : {});
      document.getElementById("compare3").hidden = true;
      plot([{ label: s.mode, series: s.series, color: modeColor(s.mode) }]);
      renderDetail(s.series, s.mode);
    }

    if (data.exec_id) {
      const chatLink = document.getElementById("chatLink");
      chatLink.href = "/download/chat/" + encodeURIComponent(data.exec_id);
      chatLink.hidden = false;
    }

    const s2 = data.saved_to || {};
    let msg = `${data.mock ? "[MOCK] " : ""}Done. exec_id: ${data.exec_id} | Firestore: ${s2.firestore || "off"}`;
    const c = data.capture;
    if (c) {
      if (c.ok && c.download) {
        const link = document.getElementById("pcapLink");
        link.href = c.download; link.hidden = false;
        msg += ` | pcap: ${fmtBytes(c.bytes)} (${c.host})`;
        if (c.stats && Object.keys(c.stats).length) msg += ` | dropped: ${c.dropped || 0}`;
      } else {
        msg += ` | capture: ${c.error || c.note || "no packets"}`;
      }
    }
    status.textContent = msg;
    loadHistory();
  } catch (e) {
    status.textContent = "Failed: " + e;
  } finally {
    btn.disabled = false;
  }
}

// --- Execution history viewer ------------------------------------------------
let histRuns = [];

async function loadHistory() {
  const resp = await fetch("/history");
  const data = await resp.json();
  histRuns = data.runs || [];
  document.getElementById("histSource").textContent =
    `source: ${data.source}` + (data.dummy ? " — showing DUMMY data (no history found)" : "");

  const tb = document.querySelector("#histTable tbody");
  tb.innerHTML = "";
  const cmpA = document.getElementById("cmpA");
  const cmpB = document.getElementById("cmpB");
  cmpA.length = 1; cmpB.length = 1;
  for (const r of histRuns) {
    const flags = (r.dummy ? "DUMMY " : "") + (r.mock ? "MOCK" : "");
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.exec_id}</td><td>${r.mode || ""}</td><td>${r.timestamp || ""}</td>
      <td>${(r.totals?.tokens ?? "").toLocaleString?.() ?? ""}</td>
      <td>${r.totals?.wire_bytes != null ? fmtBytes(r.totals.wire_bytes) : ""}</td>
      <td>${flags}</td><td><button data-id="${r.exec_id}" class="viewBtn">view</button>
      <button data-id="${r.exec_id}" class="delBtn"${r.dummy ? " disabled title='dummy row'" : ""}>🗑 delete</button></td>`;
    tb.appendChild(tr);
    for (const sel of [cmpA, cmpB]) {
      const o = document.createElement("option");
      o.value = r.exec_id; o.textContent = `${r.mode || "?"} · ${r.exec_id}`;
      sel.appendChild(o);
    }
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

async function fetchExec(execId) {
  const resp = await fetch("/history/" + encodeURIComponent(execId));
  if (!resp.ok) return null;
  return resp.json();
}

async function viewExec(execId) {
  const doc = await fetchExec(execId);
  if (!doc) return;
  document.getElementById("histDetailMeta").textContent =
    `${doc.exec_id} · ${doc.mode} · ${doc.timestamp}` + (doc.dummy ? " · DUMMY" : "") + (doc.mock ? " · MOCK" : "");
  const pre = document.getElementById("histDetail");
  pre.hidden = false;
  pre.textContent = JSON.stringify(doc, null, 2);
  const dl = document.getElementById("histDownload");
  dl.href = "/download/run/" + encodeURIComponent(execId);
  dl.hidden = false;
  // plot this execution from stored series
  const sm = doc.summary || {};
  if (sm.mode === "caching-3stage") {
    plot([
      { label: "stateless", series: sm.stateless_series, color: "#ff6b6b" },
      { label: "stateful (cache+Q)", series: sm.stateful_series, color: "#4dd4ac" },
    ]);
  } else if (sm.series) {
    plot([{ label: doc.mode, series: sm.series, color: modeColor(doc.mode) }]);
  }
}

async function compare() {
  const a = document.getElementById("cmpA").value;
  const b = document.getElementById("cmpB").value;
  if (!a || !b) return;
  const [da, db] = await Promise.all([fetchExec(a), fetchExec(b)]);
  const items = [];
  if (da) items.push({ label: `A: ${da.mode}`, series: da.summary.series, color: "#ff6b6b" });
  if (db) items.push({ label: `B: ${db.mode}`, series: db.summary.series, color: "#4dd4ac" });
  if (items.length) plot(items);
}

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
document.getElementById("start").addEventListener("click", start);
document.getElementById("refresh").addEventListener("click", loadHistory);
document.getElementById("compare").addEventListener("click", compare);
document.getElementById("inspect").addEventListener("click", inspect);
loadModels();
loadHistory();
