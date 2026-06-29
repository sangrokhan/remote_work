let tokenChart, byteChart;

function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
  return (n / 1048576).toFixed(2) + " MB";
}

function lineChart(ctx, title, labels, statelessData, deltaData, yLabel) {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "stateless (full resend)", data: statelessData, borderColor: "#ff6b6b", backgroundColor: "rgba(255,107,107,.1)", fill: true, tension: .2 },
        { label: "delta (new turn only)", data: deltaData, borderColor: "#4dd4ac", backgroundColor: "rgba(77,212,172,.1)", fill: true, tension: .2 },
      ],
    },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: title, color: "#e6e6e6" }, legend: { labels: { color: "#cbd5e0" } } },
      scales: {
        x: { title: { display: true, text: "turn", color: "#8b98a5" }, ticks: { color: "#8b98a5" }, grid: { color: "#2d3748" } },
        y: { title: { display: true, text: yLabel, color: "#8b98a5" }, ticks: { color: "#8b98a5" }, grid: { color: "#2d3748" } },
      },
    },
  });
}

function renderSummary(s) {
  const t = s.totals;
  const el = document.getElementById("summary");
  el.hidden = false;
  el.innerHTML = `
    <h2>Result</h2>
    <p>Stateless used <span class="big">${t.stateless_tokens.toLocaleString()}</span> tokens
       vs delta <span class="ok">${t.delta_tokens.toLocaleString()}</span> tokens.</p>
    <p><strong>Token ratio:</strong> ${t.token_ratio}× &nbsp;|&nbsp;
       <strong>Wire ratio:</strong> ${t.wire_ratio}×</p>
    <p><strong>Wire bytes:</strong> stateless ${fmtBytes(t.stateless_wire_bytes)} vs delta ${fmtBytes(t.delta_wire_bytes)}</p>
    <p><strong>Cost estimate</strong> (@ ${t.price_per_token}/tok):
       stateless $${t.stateless_cost_usd} vs delta $${t.delta_cost_usd}</p>
  `;
}

function renderDetail(records) {
  const tb = document.querySelector("#detail tbody");
  tb.innerHTML = "";
  for (const r of records) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.mode}</td><td>${r.turn}</td><td>${r.prompt_tokens}</td>
      <td>${r.resp_tokens}</td><td>${r.total_tokens}</td>
      <td>${r.wire_sent}</td><td>${r.wire_recv}</td><td>${r.error || ""}</td>`;
    tb.appendChild(tr);
  }
}

async function start() {
  const btn = document.getElementById("start");
  const status = document.getElementById("status");
  btn.disabled = true;
  status.textContent = "Running…";
  try {
    document.getElementById("pcapLink").hidden = true;
    const body = {
      turns: +document.getElementById("turns").value,
      message_chars: +document.getElementById("chars").value,
      model: document.getElementById("model").value,
      capture: document.getElementById("capture").checked,
    };
    const resp = await fetch("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) { status.textContent = "Error: " + (data.error || resp.status); return; }

    const s = data.summary;
    renderSummary(s);

    if (tokenChart) tokenChart.destroy();
    if (byteChart) byteChart.destroy();
    tokenChart = lineChart(
      document.getElementById("tokenChart"),
      "Cumulative tokens", s.stateless.turns,
      s.stateless.cum_tokens, s.delta.cum_tokens, "tokens");
    byteChart = lineChart(
      document.getElementById("byteChart"),
      "Cumulative wire bytes", s.stateless.turns,
      s.stateless.cum_wire_bytes, s.delta.cum_wire_bytes, "bytes");

    // detail: interleave by turn for readability
    renderDetail(data.summary.stateless.turns.flatMap((turn, i) => ([
      mkRec("stateless", s.stateless, i),
      mkRec("delta", s.delta, i),
    ])));

    const s2 = data.saved_to || {};
    let msg = `Done. JSON: ${s2.json || "-"} | Firestore: ${s2.firestore || "off"}`;
    const c = data.capture;
    if (c) {
      if (c.ok && c.download) {
        const link = document.getElementById("pcapLink");
        link.href = c.download;
        link.hidden = false;
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

// Build a detail-row-like object from a mode series at index i.
function mkRec(mode, series, i) {
  return {
    mode, turn: series.turns[i],
    prompt_tokens: series.per_turn_prompt_tokens[i],
    resp_tokens: series.per_turn_tokens[i] - series.per_turn_prompt_tokens[i],
    total_tokens: series.per_turn_tokens[i],
    wire_sent: "", wire_recv: series.per_turn_wire_bytes[i],
    error: "",
  };
}

async function loadHistory() {
  const resp = await fetch("/history");
  const data = await resp.json();
  document.getElementById("history").textContent = JSON.stringify(data.aggregate, null, 2);
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
      method: "POST",
      headers: { "Content-Type": "application/json" },
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

document.getElementById("start").addEventListener("click", start);
document.getElementById("refresh").addEventListener("click", loadHistory);
document.getElementById("inspect").addEventListener("click", inspect);
loadHistory();
