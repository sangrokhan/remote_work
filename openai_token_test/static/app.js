"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString();
const bytes = (n) => n >= 1024 ? `${(n / 1024).toFixed(1)} kB` : `${Math.round(n)} B`;

// Anything free-form that reaches innerHTML goes through this: tcpdump's stderr,
// python exception messages, fixture descriptions. All local and trusted today,
// but "trusted today" is how markup ends up executing tomorrow.
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let STATUS = null;
let ARMS = [];

// ---------------------------------------------------------------- status

async function loadStatus() {
  STATUS = await (await fetch("/status")).json();
  ARMS = STATUS.arms;

  const cap = STATUS.capture;
  const key = STATUS.key_present
    ? '<span class="good">key ok</span>'
    : '<span class="bad">OPENAI_API_KEY missing</span>';
  const capTxt = cap.available
    ? '<span class="good">capture ready</span>'
    : `<span class="bad">capture off</span> — ${esc(cap.reason)}`;

  $("status").innerHTML =
    `${esc(STATUS.model)} · ${esc(STATUS.host)} · ${key} · ${capTxt} · ` +
    `$${STATUS.prices.input}/$${STATUS.prices.cached_input}/$${STATUS.prices.output} per MTok`;

  $("capture").disabled = !cap.available;
  $("capture").title = cap.available ? "" : cap.reason;

  const sel = $("fixture");
  sel.innerHTML = "";
  for (const [name, f] of Object.entries(STATUS.fixtures)) {
    const o = document.createElement("option");
    o.value = name;
    // textContent, so the fixture's free-form description can never be markup
    o.textContent = `${name} — ${f.turns} turns, ${fmt(f.system_chars)}-char system prompt`;
    o.title = f.description || "";
    sel.appendChild(o);
  }
  sel.onchange = clampTurns;
  clampTurns();

  $("arms").innerHTML = ARMS
    .map((a) => `<span class="arm-chip ${a}">${a}</span>`)
    .join("");

  renderHint();
  loadHistory();
}

function clampTurns() {
  const f = STATUS.fixtures[$("fixture").value];
  if (!f) return;
  $("turns").max = f.turns;
  if (+$("turns").value > f.turns) $("turns").value = f.turns;
  renderHint();
}

function renderHint() {
  const turns = +$("turns").value, repeats = +$("repeats").value;
  const f = STATUS.fixtures[$("fixture").value];
  if (!f) return;
  // system prompt resent every turn by the two stateless arms; ~4 chars/token
  const sysTok = f.system_chars / 4;
  const statelessTok = 2 * repeats * (sysTok * turns + (turns * (turns - 1)) / 2 * 300);
  const statefulTok = repeats * (sysTok * turns);
  const est = ((statelessTok + statefulTok) / 1e6) * STATUS.prices.input;
  $("cost-hint").textContent =
    `${ARMS.length} arms × ${turns} turns × ${repeats} repeat(s) = ` +
    `${ARMS.length * turns * repeats} model calls. Very roughly $${est.toFixed(3)} at list price.`;
}
["turns", "repeats"].forEach((id) =>
  document.addEventListener("input", (e) => { if (e.target.id === id) renderHint(); })
);

// ---------------------------------------------------------------- run

const progressState = new Map();

function progressRow(arm) {
  if (!progressState.has(arm)) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML =
      `<span class="bar-label ${arm}">${arm}</span>` +
      `<span class="bar"><i></i></span>` +
      `<span class="bar-num"></span>`;
    $("progress").appendChild(row);
    const color = getComputedStyle(document.documentElement)
      .getPropertyValue(arm === "responses_stateful" ? "--stateful"
        : arm === "responses_stateless" ? "--stateless2" : "--stateless");
    row.querySelector("i").style.background = color;
    progressState.set(arm, { row, uploaded: 0 });
  }
  return progressState.get(arm);
}

function log(line) {
  const el = $("log");
  el.textContent += line + "\n";
  el.scrollTop = el.scrollHeight;
}

$("start").onclick = async () => {
  $("start").disabled = true;
  $("progress").innerHTML = "";
  $("log").textContent = "";
  progressState.clear();
  $("progress-panel").hidden = false;
  $("result-panel").hidden = true;

  const body = {
    fixture: $("fixture").value,
    turns: +$("turns").value,
    repeats: +$("repeats").value,
    capture: $("capture").checked,
    stream: $("stream").checked,
  };

  const resp = await fetch("/run/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  // SSE over a POST body: read the stream by hand, EventSource cannot POST.
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });

    let sep;
    while ((sep = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      const ev = /^event: (.+)$/m.exec(chunk)?.[1];
      const data = JSON.parse(/^data: (.+)$/m.exec(chunk)?.[1] || "null");
      handle(ev, data);
    }
  }
  $("start").disabled = false;
};

function handle(ev, d) {
  if (ev === "start") {
    log(`start · ${d.model} · ${d.turns} turns × ${d.repeats} · ${d.arms.join(", ")}` +
        (d.stream ? " · streaming (TTFT/TTLT)" : ""));
  } else if (ev === "turn") {
    const st = progressRow(d.arm);
    st.uploaded += d.upload_bytes;
    const pct = (d.turn / d.turns) * 100;
    st.row.querySelector("i").style.width = `${pct}%`;
    st.row.querySelector(".bar-num").textContent =
      `${d.turn}/${d.turns} · ${bytes(st.uploaded)}`;
    const timing = d.streamed
      ? `ttft=${String(d.ttft_ms).padStart(5)}ms ttlt=${String(d.ttlt_ms).padStart(6)}ms`
      : `${String(d.latency_ms).padStart(6)}ms`;
    log(
      `${d.arm.padEnd(20)} turn ${String(d.turn).padStart(2)}/${d.turns}  ` +
      `up=${String(d.upload_bytes).padStart(7)}B  ` +
      `in=${String(d.input_tokens).padStart(6)} (cached ${String(d.cached_tokens).padStart(6)})  ` +
      timing
    );
  } else if (ev === "note") {
    log(`note: ${d.message}`);
  } else if (ev === "error") {
    log(`ERROR: ${d.message}`);
  } else if (ev === "done") {
    log(`done · saved as ${d.exec_id}`);
    renderResult(d);
    loadHistory();
  }
}

// ---------------------------------------------------------------- result

function renderResult(d) {
  $("result-panel").hidden = false;
  const s = d.summary;
  const arms = s.arms;

  // headline: the one comparison the whole thing exists to make
  const r = Object.values(s.ratios)[0];
  if (r) {
    $("headline").innerHTML =
      `<div>Client uploads <b class="big">${r.upload_bytes.toFixed(1)}×</b> more bytes ` +
      `when it holds the history itself.</div>` +
      `<div>It is billed for <b class="big">${r.input_tokens.toFixed(2)}×</b> the input tokens. ` +
      `That is the gap: <b>the bytes are savable, the billing is not.</b></div>`;
  }

  const cols = [
    ["arm", (a, t) => a],
    ["upload B", (a, t) => fmt(t.req_payload_bytes)],
    ["wire sent", (a, t) => fmt(t.wire_sent)],
    ["in tok", (a, t) => fmt(t.input_tokens)],
    ["cached", (a, t) => fmt(t.cached_tokens)],
    ["billed", (a, t) => fmt(t.billed_uncached_tokens)],
    ["out tok", (a, t) => fmt(t.output_tokens)],
    ["cost $", (a, t) => t.cost_usd.toFixed(5)],
    ["mean ms", (a, t) => fmt(t.mean_latency_ms)],
  ];
  $("totals").innerHTML =
    `<tr>${cols.map((c) => `<th>${c[0]}</th>`).join("")}</tr>` +
    Object.entries(arms).map(([a, v]) =>
      `<tr class="${a}">${cols.map((c) => `<td class="num">${c[1](a, v.totals)}</td>`).join("")}</tr>`
    ).join("");

  // TTFT/TTLT only exist when the run streamed
  const streamed = Object.values(arms).some((v) => v.streamed);
  $("latency-block").hidden = !streamed;
  if (streamed) {
    const lat = [
      ["arm", (a, v) => a],
      ["TTFT mean", (a, v) => fmt(v.latency.ttft_ms.mean)],
      ["TTFT p50", (a, v) => fmt(v.latency.ttft_ms.p50)],
      ["TTFT p95", (a, v) => fmt(v.latency.ttft_ms.p95)],
      ["TTLT mean", (a, v) => fmt(v.latency.ttlt_ms.mean)],
      ["TTLT p50", (a, v) => fmt(v.latency.ttlt_ms.p50)],
      ["TTLT p95", (a, v) => fmt(v.latency.ttlt_ms.p95)],
    ];
    $("latency").innerHTML =
      `<tr>${lat.map((c) => `<th>${c[0]}</th>`).join("")}</tr>` +
      Object.entries(arms).map(([a, v]) =>
        `<tr class="${a}">${lat.map((c) => `<td class="num">${c[1](a, v)}</td>`).join("")}</tr>`
      ).join("");
  }

  // per-turn upload, the shape of the argument
  const n = Object.values(arms)[0].turns;
  const head = ["arm", ...Array.from({ length: n }, (_, i) => `t${i + 1}`)];
  $("perturn").innerHTML =
    `<tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr>` +
    Object.entries(arms).map(([a, v]) =>
      `<tr class="${a}"><td>${a}</td>` +
      v.per_turn.req_payload_bytes.map((b) => `<td class="num">${bytes(b)}</td>`).join("") +
      `</tr>`
    ).join("");

  const img = $("charts");
  img.src = `/download/${d.exec_id}/charts.png?t=${Date.now()}`;
  img.hidden = false;

  const caps = d.captures || [];
  $("captures").innerHTML = caps.length
    ? caps.map((c) => c.ok
      ? `<div>pcap ${esc(c.arm)} r${c.repeat}: ${bytes(c.bytes)}, ` +
        `${c.stats?.captured ?? "?"} packets` +
        (c.dropped ? ` <span class="drop">· ${c.dropped} dropped</span>` : "") +
        ` · <a href="/download/pcap/${encodeURIComponent(c.file)}">download</a></div>`
      : `<div class="drop">pcap ${esc(c.arm)}: ${esc(c.error || c.note)}</div>`).join("")
    : "";

  $("downloads").innerHTML = [
    ["run.json", "run.json"],
    ["summary.csv", "summary.csv"],
    ["charts.png", "charts.png"],
    [`bodies.zip (${d.manifest.bodies} files)`, "bodies.zip"],
  ].map(([label, what]) =>
    `<a href="/download/${d.exec_id}/${what}">${label}</a>`
  ).join("");
}

// ---------------------------------------------------------------- history

async function loadHistory() {
  const { runs } = await (await fetch("/history")).json();
  if (!runs.length) {
    $("history").innerHTML = "<tr><td>no runs yet</td></tr>";
    return;
  }
  $("history").innerHTML =
    "<tr><th>when</th><th>model</th><th>fixture</th><th>turns</th>" +
    "<th>repeats</th><th>upload ×</th><th>token ×</th><th>pcaps</th><th></th></tr>" +
    runs.map((r) => {
      const id = encodeURIComponent(r.exec_id);
      return `
      <tr>
        <td>${esc(r.timestamp.slice(0, 19).replace("T", " "))}</td>
        <td class="num">${esc(r.model)}</td>
        <td class="num">${esc(r.fixture)}</td>
        <td class="num">${r.turns}</td>
        <td class="num">${r.repeats}</td>
        <td class="num">${(r.upload_ratio || 0).toFixed(1)}×</td>
        <td class="num">${(r.token_ratio || 0).toFixed(2)}×</td>
        <td class="num">${r.captures || ""}</td>
        <td><a href="/download/${id}/run.json">json</a>
            · <a href="/download/${id}/summary.csv">csv</a>
            · <a href="/download/${id}/bodies.zip">bodies</a></td>
      </tr>`;
    }).join("");
}

loadStatus();
