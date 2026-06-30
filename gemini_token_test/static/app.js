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

async function start() {
  const btn = document.getElementById("start");
  const status = document.getElementById("status");
  btn.disabled = true;
  status.textContent = "Running…";
  try {
    document.getElementById("pcapLink").hidden = true;
    const body = {
      mode: document.getElementById("mode").value,
      turns: +document.getElementById("turns").value,
      model: selectedModel(),
      capture: document.getElementById("capture").checked,
    };
    const resp = await fetch("/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) { status.textContent = "Error: " + (data.error || resp.status); return; }

    const s = data.summary;
    renderSummary(s.totals, data.mock, false);
    plot([{ label: s.mode, series: s.series, color: modeColor(s.mode) }]);
    renderDetail(s.series, s.mode);

    const s2 = data.saved_to || {};
    let msg = `${data.mock ? "[MOCK] " : ""}Done. exec_id: ${data.exec_id} | Firestore: ${s2.firestore || "off"}`;
    const c = data.capture;
    if (c) {
      if (c.ok && c.download) {
        const link = document.getElementById("pcapLink");
        link.href = c.download; link.hidden = false;
        msg += ` | pcap: ${fmtBytes(c.bytes)} (${c.host})`;
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
      <td>${flags}</td><td><button data-id="${r.exec_id}" class="viewBtn">view</button></td>`;
    tb.appendChild(tr);
    for (const sel of [cmpA, cmpB]) {
      const o = document.createElement("option");
      o.value = r.exec_id; o.textContent = `${r.mode || "?"} · ${r.exec_id}`;
      sel.appendChild(o);
    }
  }
  document.querySelectorAll(".viewBtn").forEach(b =>
    b.addEventListener("click", () => viewExec(b.dataset.id)));
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
  plot([{ label: doc.mode, series: doc.summary.series, color: modeColor(doc.mode) }]);
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
