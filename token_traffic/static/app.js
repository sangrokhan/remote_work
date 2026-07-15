/* token_traffic — the operator's console.
 *
 * Three rules this file exists to keep, and everything else is plumbing:
 *
 *   1. A mock run must never be able to pass for a measured one. Not in the page,
 *      not in a screenshot of it: the banner is sticky, every result carries the
 *      badge, and every chart drawn from synthetic data is watermarked in the
 *      canvas itself, so a cropped PNG still says MOCK.
 *   2. Money is spent only on an explicit second click. The preflight is what the
 *      operator confirms, not the run button.
 *   3. Prep is never folded into an arm's totals, and a failure is never absent.
 *      The server already refuses to mix them; the UI must not re-mix them.
 *
 * No CDN, no external asset. Charts are inline canvas 2D.
 */

'use strict';

const $ = (id) => document.getElementById(id);

const state = {
  config: null,
  selection: {},          // provider -> Set(arm)
  preflight: null,        // last /api/preflight response, for the selection below
  confirmed: false,
  running: false,
  run: null,              // the run document currently on screen
};

// One colour per series. Chosen to stay distinguishable on a dark panel and in
// greyscale print; the swatch in the legend is the only key the charts have.
const PALETTE = ['#5aa9ff', '#7ee2a8', '#ffb454', '#ff8fa3', '#c792ea',
                 '#4fd1c5', '#f6e05e', '#fc8181', '#90cdf4', '#b5e853'];
const colorOf = (keys, k) => PALETTE[Math.max(0, keys.indexOf(k)) % PALETTE.length];

const fmtInt = (n) => (n === null || n === undefined || n === '') ? '—'
  : Number(n).toLocaleString('en-US');
const fmtBytes = (n) => {
  const v = Number(n || 0);
  if (v >= 1048576) return (v / 1048576).toFixed(2) + ' MB';
  if (v >= 1024) return (v / 1024).toFixed(1) + ' KB';
  return v + ' B';
};
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

/* ------------------------------------------------------------------ config */

async function loadConfig() {
  const cfg = await (await fetch('/api/config')).json();
  state.config = cfg;

  if (cfg.mock) {
    $('mockBanner').hidden = false;
    document.body.classList.add('is-mock');
    document.title = '[MOCK] ' + document.title;
  } else {
    // Out of the DOM, not merely hidden: a banner that survives a live run as an
    // invisible node is one CSS rule away from being visible in every run, which is
    // exactly what happened -- `display: flex` on the class beat the [hidden]
    // attribute, and every live run wore a MOCK banner until somebody noticed.
    $('mockBanner').remove();
  }

  renderConfig(cfg);
  renderPicker(cfg);

  const m = $('measure');
  clear(m);
  for (const name of cfg.measures) {
    const o = el('option', null, name);
    o.value = name;
    if (name === 'bytes') o.selected = true;   // the cheap default: one call per turn
    m.appendChild(o);
  }

  const f = $('fixture');
  clear(f);
  for (const name of cfg.fixtures) {
    const o = el('option', null, name);
    o.value = name;
    if (name === cfg.fixture.name) o.selected = true;
    f.appendChild(o);
  }

  const t = $('turns');
  t.value = cfg.fixture.turns;
  t.max = cfg.fixture.turns;
}

function renderConfig(cfg) {
  const p = $('cfgProviders');
  clear(p);
  for (const prov of cfg.providers) {
    const row = el('div', 'row');
    const badge = el('span', 'badge ' + (prov.ready ? 'ok' : 'bad'),
                     prov.ready ? 'READY' : 'NOT READY');
    row.appendChild(badge);
    row.appendChild(document.createTextNode(' '));
    const b = el('b', null, prov.name);
    row.appendChild(b);
    row.appendChild(document.createTextNode(' ' + prov.model));
    row.appendChild(el('span', 'why', prov.reason || ''));
    row.appendChild(el('span', 'why', prov.arms.length + ' arms, ' +
                       prov.headline_arms.length + ' headline'));
    p.appendChild(row);
  }

  const c = $('cfgCapture');
  clear(c);
  const crow = el('div', 'row');
  crow.appendChild(el('span', 'badge ' + (cfg.capture.available ? 'ok' : 'warn'),
                      cfg.capture.available ? 'AVAILABLE' : 'UNAVAILABLE'));
  crow.appendChild(el('span', 'why', cfg.capture.reason || ''));
  crow.appendChild(el('span', 'why', 'dir: ' + cfg.capture.dir));
  c.appendChild(crow);
  if (!cfg.capture.available) $('capture').disabled = true;

  const x = $('cfgFixture');
  clear(x);
  const r1 = el('div', 'row');
  r1.appendChild(el('b', null, cfg.fixture.name));
  r1.appendChild(document.createTextNode(' — ' + cfg.fixture.turns + ' turns'));
  r1.appendChild(el('span', 'why', cfg.fixture.description || ''));
  x.appendChild(r1);
  const r2 = el('div', 'row');
  r2.appendChild(document.createTextNode('retention: keep ' + cfg.retention_keep +
                                         ' runs per bucket'));
  r2.appendChild(el('span', 'why', cfg.mock
    ? 'MOCK: runs are stored in the mock bucket and never listed with live runs.'
    : 'live: runs reach a paid API.'));
  x.appendChild(r2);
}

/* --------------------------------------------------------------- selection */

function renderPicker(cfg) {
  const host = $('armPicker');
  clear(host);
  state.selection = {};

  for (const prov of cfg.providers) {
    const box = el('div', 'prov' + (prov.ready ? '' : ' notready'));
    const head = el('div', 'prov-head');
    head.appendChild(el('b', null, prov.name));
    head.appendChild(el('span', 'prov-model', prov.model));
    box.appendChild(head);
    if (!prov.ready) box.appendChild(el('div', 'why', prov.reason || 'not ready'));

    const chosen = new Set();
    for (const arm of prov.arms) {
      const headline = prov.headline_arms.includes(arm);
      const lab = el('label', 'arm');
      const cb = el('input');
      cb.type = 'checkbox';
      cb.value = arm;
      cb.disabled = !prov.ready;
      cb.checked = prov.ready && headline;      // default = HEADLINE_ARMS
      if (cb.checked) chosen.add(arm);
      cb.addEventListener('change', () => {
        if (cb.checked) chosen.add(arm); else chosen.delete(arm);
        invalidatePreflight();
      });
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(arm));
      if (!headline) lab.appendChild(el('span', 'hl', '(diagnostic)'));
      box.appendChild(lab);
    }
    state.selection[prov.name] = chosen;
    host.appendChild(box);
  }
}

function selectionPayload() {
  const providers = {};
  for (const [name, arms] of Object.entries(state.selection)) {
    if (arms.size) providers[name] = [...arms];
  }
  const turns = $('turns').value;
  return {
    providers,
    measure: $('measure').value,
    turns: turns === '' ? null : Number(turns),
    fixture: $('fixture').value,
    capture: $('capture').checked,
    cache_bust: $('cacheBust').checked,
    pause_seconds: Number($('pause').value || 0),
  };
}

// Any change to the selection makes the last preflight a lie about what would run,
// so the run button goes back behind a fresh preflight.
function invalidatePreflight() {
  state.preflight = null;
  state.confirmed = false;
  $('runBtn').disabled = true;
  $('runBtn').classList.remove('billable');
  $('preflightOut').hidden = true;
}
for (const id of ['measure', 'turns', 'fixture', 'capture', 'cacheBust', 'pause']) {
  // listener attached after DOM parse; ids exist because the script is at body end
  $(id).addEventListener('change', invalidatePreflight);
}

/* --------------------------------------------------------------- preflight */

async function doPreflight() {
  const sel = selectionPayload();
  if (!Object.keys(sel.providers).length) {
    showPreflightError('Nothing selected. Pick at least one arm.');
    return;
  }
  const res = await fetch('/api/preflight', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(sel),
  });
  const pf = await res.json();
  if (!pf.ok) { showPreflightError(pf.error || 'preflight failed'); return; }

  state.preflight = pf;
  state.confirmed = false;
  renderPreflight(pf);
}

function showPreflightError(msg) {
  const out = $('preflightOut');
  out.hidden = false;
  out.className = 'preflight billable';
  clear(out);
  out.appendChild(el('div', null, 'preflight: ' + msg));
  $('runBtn').disabled = true;
}

function renderPreflight(pf) {
  const out = $('preflightOut');
  const billable = Number(pf.billable_calls || 0);
  out.hidden = false;
  out.className = 'preflight' + (billable > 0 ? ' billable' : '');
  clear(out);

  const head = el('div');
  head.appendChild(el('span', 'big' + (billable ? '' : ' free'), String(billable)));
  head.appendChild(document.createTextNode(
    ' billable API call' + (billable === 1 ? '' : 's') +
    ' — ' + pf.pairs.length + ' arm' + (pf.pairs.length === 1 ? '' : 's') +
    ' × ' + pf.turns + ' turns' +
    ($('measure').value === 'both' ? ' × 2 passes (measure=both)' : '')));
  out.appendChild(head);

  if (pf.mock) {
    const m = el('div');
    m.appendChild(el('span', 'badge mock', 'MOCK MODE'));
    m.appendChild(document.createTextNode(
      ' — nothing will be sent, nothing will be billed, and nothing will be measured.'));
    out.appendChild(m);
  }

  out.appendChild(el('div', null, 'arms: ' + pf.pairs.join('  ')));

  if (pf.warnings && pf.warnings.length) {
    const ul = el('ul');
    for (const w of pf.warnings) ul.appendChild(el('li', null, w));
    out.appendChild(ul);
  }

  const btn = $('runBtn');
  btn.classList.toggle('billable', billable > 0);

  if (billable > 0) {
    // Real money. The run button stays dead until this box is ticked, and the tick
    // is reset by any change to the selection.
    const conf = el('label', 'confirm');
    const cb = el('input');
    cb.type = 'checkbox';
    cb.id = 'confirmBox';
    cb.addEventListener('change', () => {
      state.confirmed = cb.checked;
      btn.disabled = !cb.checked || state.running;
    });
    conf.appendChild(cb);
    conf.appendChild(document.createTextNode(
      'I understand this will make ' + billable + ' billable call' +
      (billable === 1 ? '' : 's') + ' against a paid API.'));
    out.appendChild(conf);
    btn.disabled = true;
  } else {
    state.confirmed = true;
    btn.disabled = state.running;
  }
}

/* --------------------------------------------------------------------- run */

async function doRun() {
  if (!state.preflight || !state.confirmed) return;
  state.running = true;
  $('runBtn').disabled = true;
  $('preflightBtn').disabled = true;

  const pf = state.preflight;
  startProgress(pf);

  const res = await fetch('/api/run/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(selectionPayload()),
  });

  if (!res.ok || !res.body) {
    let msg = 'HTTP ' + res.status;
    try { msg = (await res.json()).error || msg; } catch (_) { /* not JSON */ }
    finishProgress('failed: ' + msg, true);
    endRun();
    return;
  }

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let done = false;

  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buf += dec.decode(chunk.value, {stream: true});

    let cut;
    while ((cut = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      for (const line of frame.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        let ev;
        try { ev = JSON.parse(line.slice(6)); } catch (_) { continue; }

        if (ev.event === 'done') {
          done = true;
          finishProgress('complete', false);
          showRun(ev.run);
          loadHistory();
        } else if (ev.event === 'error') {
          done = true;
          finishProgress(ev.error || 'error', true);
        } else {
          onProgress(ev);
        }
      }
    }
  }
  // The stream closing without a terminal event means the run died on the far side;
  // silence here would look exactly like success.
  if (!done) finishProgress('stream ended without a result', true);
  endRun();
}

function endRun() {
  state.running = false;
  $('preflightBtn').disabled = false;
  invalidatePreflight();       // the next run needs its own preflight
}

/* ---------------------------------------------------------------- progress */

const prows = {};   // "provider:arm" -> {row, bar, phase, n}

function startProgress(pf) {
  const host = $('progressRows');
  clear(host);
  for (const k in prows) delete prows[k];
  $('progressPanel').hidden = false;
  $('progressState').textContent = 'starting…';

  for (const key of pf.pairs) {
    const row = el('div', 'prow');
    row.appendChild(el('span', 'k', key));
    const ph = el('span', 'ph', 'queued');
    row.appendChild(ph);
    const bar = el('div', 'bar');
    const fill = el('i');
    fill.style.width = '0%';
    bar.appendChild(fill);
    row.appendChild(bar);
    const n = el('span', 'n', '0/' + pf.turns);
    row.appendChild(n);
    host.appendChild(row);
    prows[key] = {row, ph, fill, n, turns: pf.turns};
  }
}

function onProgress(ev) {
  const key = ev.provider + ':' + ev.arm;
  const r = prows[key];

  if (ev.phase === 'pause') {
    $('progressState').textContent =
      'pausing ' + ev.remaining + 's/' + ev.pause_total + 's before ' + key;
    if (r) r.ph.textContent = 'pausing';
    return;
  }
  if (!r) return;

  $('progressState').textContent = key + ' · ' + ev.phase;
  r.ph.textContent = ev.phase;
  const turns = ev.turns || r.turns || 1;
  const pct = Math.min(100, Math.round(((ev.turn || 0) / turns) * 100));
  r.fill.style.width = pct + '%';
  r.n.textContent = (ev.turn || 0) + '/' + turns;
  if ((ev.turn || 0) >= turns) r.row.classList.add('done');
}

function finishProgress(msg, bad) {
  const s = $('progressState');
  s.textContent = msg;
  s.style.color = bad ? 'var(--bad)' : 'var(--ok)';
}

/* ----------------------------------------------------------------- results */

function showRun(run) {
  state.run = run;
  const s = run.summary || {};
  const keys = s.keys || [];
  const isMock = !!run.mock;

  $('resultsPanel').hidden = false;

  const badge = $('resultBadge');
  badge.className = 'badge ' + (isMock ? 'mock' : 'live');
  badge.textContent = isMock ? 'MOCK — SYNTHETIC, NOT MEASURED' : 'LIVE';

  const meta = $('resultMeta');
  clear(meta);
  const p = run.params || {};
  const bits = [
    ['exec_id', run.exec_id || '—'],
    ['when', run.timestamp || '—'],
    ['measure', s.measure || p.measure || '—'],
    ['fixture', p.fixture || '—'],
    ['turns', p.turns],
    ['capture', p.capture ? 'on' : 'off'],
    ['models', Object.entries(p.models || {}).map(([k, v]) => k + '=' + v).join(' ')],
  ];
  for (const [k, v] of bits) {
    const row = el('div', 'row');
    row.appendChild(el('b', null, k + ': '));
    row.appendChild(document.createTextNode(String(v === undefined ? '—' : v)));
    meta.appendChild(row);
  }

  // Warnings from the runner (a capture that could not start, a `both` pass that
  // would double-bill) belong on the result, not only on the preflight.
  const warns = $('resultWarnings');
  clear(warns);
  if ((p.warnings || []).length) {
    const box = el('div', 'alert warn');
    box.appendChild(el('b', null, 'warnings'));
    const ul = el('ul');
    for (const w of p.warnings) ul.appendChild(el('li', null, w));
    box.appendChild(ul);
    warns.appendChild(box);
  }

  // A failed call produces a number shaped exactly like a good one. Name every one.
  const fails = $('resultFailures');
  clear(fails);
  const failures = s.failures || [];
  if (failures.length) {
    const box = el('div', 'alert fail');
    box.appendChild(el('b', null, failures.length + ' failed call' +
                       (failures.length === 1 ? '' : 's') +
                       ' — the numbers below are missing these turns'));
    const ul = el('ul');
    for (const f of failures) {
      ul.appendChild(el('li', null,
        f.key + ' · phase=' + f.phase + ' · turn=' + f.turn + ' · ' + f.error));
    }
    box.appendChild(ul);
    fails.appendChild(box);
  }

  drawCharts(s, keys, isMock);
  renderLegend(keys);
  renderTotals(s, keys);
  renderMarks(s, keys);
  renderPrep(s, keys);
  renderDownloads(run);

  $('resultsPanel').scrollIntoView({behavior: 'smooth', block: 'start'});
}

function seriesFor(s, keys, field) {
  return keys.map((k) => {
    const ser = s.series[k] || {};
    return {
      label: k,
      color: colorOf(keys, k),
      x: ser.turns || [],
      y: ser[field] || [],
    };
  }).filter((d) => d.y.length);
}

function drawCharts(s, keys, isMock) {
  lineChart($('chartUp'), seriesFor(s, keys, 'cum_wire_sent'), 'bytes', isMock, fmtBytes);
  lineChart($('chartDown'), seriesFor(s, keys, 'cum_wire_recv'), 'bytes', isMock, fmtBytes);
  lineChart($('chartTok'), seriesFor(s, keys, 'cum_input_tokens'), 'tokens', isMock, fmtInt);
  lineChart($('chartTail'), seriesFor(s, keys, 'per_turn_store_tail_ms'), 'ms', isMock, fmtInt);
}

function renderLegend(keys) {
  const host = $('chartLegend');
  clear(host);
  for (const k of keys) {
    const item = el('span');
    const sw = el('span', 'swatch');
    sw.style.background = colorOf(keys, k);
    item.appendChild(sw);
    item.appendChild(document.createTextNode(k));
    host.appendChild(item);
  }
}

/* ------------------------------------------------------------------ tables */

function table(node, head, rows) {
  clear(node);
  const thead = el('thead');
  const htr = el('tr');
  for (const h of head) htr.appendChild(el('th', null, h));
  thead.appendChild(htr);
  node.appendChild(thead);

  const tbody = el('tbody');
  if (!rows.length) {
    const tr = el('tr', 'empty');
    const td = el('td', null, 'nothing to show');
    td.colSpan = head.length;
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
  for (const r of rows) {
    const tr = el('tr');
    r.forEach((cell, i) => {
      const td = el('td', i === 0 ? null : 'num');
      if (i === 0 && typeof cell === 'object') {
        const sw = el('span', 'swatch');
        sw.style.background = cell.color;
        td.appendChild(sw);
        td.appendChild(document.createTextNode(cell.text));
      } else {
        td.textContent = String(cell);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  node.appendChild(tbody);
}

function renderTotals(s, keys) {
  const rows = keys.map((k) => {
    const t = s.totals[k] || {};
    return [
      {text: k, color: colorOf(keys, k)},
      t.measure || '—',
      fmtInt(t.turns),
      fmtBytes(t.wire_sent),
      fmtBytes(t.wire_recv),
      fmtBytes(t.wire),
      fmtInt(t.input_tokens),
      fmtInt(t.cached_tokens),
      fmtInt(t.output_tokens),
      fmtInt(t.reasoning_tokens),
      fmtInt(t.call_ms),
      fmtInt(t.wall_ms),
      t.errors ? String(t.errors) : '0',
    ];
  });
  table($('totalsTable'),
    ['arm', 'measure', 'turns', 'uplink', 'downlink', 'total wire', 'input tok',
     'cached tok', 'output tok', 'reasoning tok', 'call ms', 'wall ms', 'errors'],
    rows);
}

const MARKS = ['req_sent_ms', 'ttfb_ms', 'ttft_ms', 'ttlt_ms', 'turn_end_ms',
               'store_tail_ms'];

function renderMarks(s, keys) {
  const rows = keys.map((k) => {
    const t = s.totals[k] || {};
    const m = t.marks || {};
    const cells = [{text: k, color: colorOf(keys, k)}];
    for (const name of MARKS) {
      const st = m[name] || {};
      cells.push(st.n ? fmtInt(st.median) + ' / ' + fmtInt(st.max) : '—');
    }
    return cells;
  });
  table($('marksTable'), ['arm', ...MARKS], rows);
}

// One row per (arm, kind), never one per arm. A prep phase mixes calls that do not
// measure the same thing -- a Gemini transcript call is real inference with real input
// tokens, a cache build's only number is a size, a conversation create runs no inference
// and is billed nothing. Rolled into one row, their token columns got added together and
// produced a number describing nothing. So an unbilled kind shows no token count at all
// and says why in its own words: a zero there reads as "free", and it is not.
function renderPrep(s, keys) {
  const prep = s.prep || {};
  const rows = [];
  for (const k of Object.keys(prep)) {
    const p = prep[k];
    for (const b of p.by_kind || []) {
      rows.push([
        {text: k, color: colorOf(keys, k)},
        b.kind,
        fmtInt(b.calls),
        fmtBytes(b.wire_sent),
        fmtBytes(b.wire_recv),
        b.billed ? fmtInt(b.input_tokens) : '—',
        b.billed ? fmtInt(b.output_tokens) : '—',
        b.cache_tokens ? fmtInt(b.cache_tokens) : '—',
        fmtInt(b.elapsed_ms),
        b.note || '',
      ]);
    }
  }
  table($('prepTable'),
    ['arm', 'kind', 'calls', 'uplink', 'downlink', 'input tok billed',
     'output tok billed', 'cache size tok', 'elapsed ms', 'what it is'],
    rows);
}

function renderDownloads(run) {
  const host = $('downloads');
  clear(host);
  const id = run.exec_id;
  if (!id) return;

  for (const [label, href] of [
    ['records.csv', `/api/runs/${id}/records.csv`],
    ['summary.csv', `/api/runs/${id}/summary.csv`],
  ]) {
    const a = el('a', 'dl', (run.mock ? 'MOCK ' : '') + label);
    a.href = href;
    a.setAttribute('download', '');
    host.appendChild(a);
  }

  // The run document itself, not the {ok, run} envelope GET /api/runs/<id> returns —
  // and, like the CSVs, a mock run says so in the filename it lands under.
  const tag = run.mock ? 'mock_' : '';
  const blob = new Blob([JSON.stringify(run, null, 2)], {type: 'application/json'});
  const j = el('a', 'dl', (run.mock ? 'MOCK ' : '') + 'run.json');
  j.href = URL.createObjectURL(blob);
  j.download = `${tag}run_${id}.json`;
  host.appendChild(j);

  const pl = $('pcapList');
  clear(pl);
  const pcaps = run.pcaps || {};
  const names = Object.keys(pcaps);
  if (!names.length) {
    if (run.params && run.params.capture) {
      pl.appendChild(el('span', 'note', 'capture was on but produced no pcap.'));
    }
    return;
  }
  // pcaps is {key: {kind: result}} — one entry per captured pass. A single-measure run
  // has one kind; a `both` run captures the blocking and streamed passes into separate
  // files, so its arm shows a bytes link and a latency link side by side.
  for (const key of names) {
    const byKind = pcaps[key] || {};
    for (const kind of Object.keys(byKind)) {
      const c = byKind[kind] || {};
      const label = `pcap ${key} · ${kind}`;
      if (c.ok && c.file) {
        const a = el('a', 'dl', `${label} · ${fmtBytes(c.bytes)}` +
                                (c.dropped ? ` · ${c.dropped} dropped` : ''));
        a.href = '/api/pcaps/' + encodeURIComponent(c.file);
        a.setAttribute('download', '');
        pl.appendChild(a);
      } else {
        pl.appendChild(el('span', 'dl dead',
          `${label} failed: ${c.error || c.note || 'empty capture'}`));
      }
    }
  }
}

/* ------------------------------------------------------------------ charts */

/* A line chart, drawn by hand. No library, no CDN: the page has to work with the
 * network unplugged, which is the same page an operator uses to explain why a
 * number is what it is. */
function lineChart(canvas, series, unit, isMock, fmt) {
  const cssW = canvas.clientWidth || 640;
  const cssH = 300;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = {l: 62, r: 12, t: 12, b: 26};
  const w = cssW - pad.l - pad.r;
  const h = cssH - pad.t - pad.b;

  const xs = series.flatMap((s) => s.x);
  const ys = series.flatMap((s) => s.y);
  if (!xs.length || !ys.length) {
    ctx.fillStyle = '#8b98a5';
    ctx.font = '12px ui-monospace, monospace';
    ctx.fillText('no data', pad.l, pad.t + 20);
    if (isMock) watermark(ctx, cssW, cssH);
    return;
  }

  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMax = Math.max(...ys, 1);
  const X = (v) => pad.l + (xMax === xMin ? w / 2 : ((v - xMin) / (xMax - xMin)) * w);
  const Y = (v) => pad.t + h - (v / yMax) * h;

  // grid + y axis
  ctx.strokeStyle = '#222a35';
  ctx.fillStyle = '#8b98a5';
  ctx.lineWidth = 1;
  ctx.font = '10px ui-monospace, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const v = (yMax / 4) * i;
    const y = Math.round(Y(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + w, y);
    ctx.stroke();
    ctx.fillText(fmt(Math.round(v)), pad.l - 6, y);
  }

  // x axis: one tick per turn, thinned when there are many
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const ticks = [...new Set(xs)].sort((a, b) => a - b);
  const step = Math.max(1, Math.ceil(ticks.length / 12));
  ticks.forEach((t, i) => {
    if (i % step) return;
    ctx.fillText(String(t), X(t), pad.t + h + 6);
  });
  ctx.fillStyle = '#8b98a5';
  ctx.textAlign = 'left';
  ctx.fillText('turn', pad.l, pad.t + h + 16);

  // series
  ctx.lineWidth = 2;
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.beginPath();
    s.y.forEach((v, i) => {
      const x = X(s.x[i]), y = Y(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    s.y.forEach((v, i) => {
      ctx.beginPath();
      ctx.arc(X(s.x[i]), Y(v), 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  ctx.fillStyle = '#8b98a5';
  ctx.textAlign = 'right';
  ctx.fillText(unit, cssW - pad.r, pad.t + h + 6);

  if (isMock) watermark(ctx, cssW, cssH);
}

/* Burned into the pixels, not into the page around them: a screenshot of a mock
 * chart still says so once it is out of this browser. */
function watermark(ctx, w, h) {
  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.rotate(-Math.atan2(h, w));
  ctx.font = 'bold 46px ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(255, 217, 102, 0.20)';
  ctx.fillText('MOCK — NOT MEASURED', 0, 0);
  ctx.restore();
}

/* ----------------------------------------------------------------- history */

async function loadHistory() {
  const h = await (await fetch('/api/runs')).json();
  $('historyMeta').textContent =
    `keep ${h.keep} per bucket · ${h.dir} · ${h.runs.length} live, ${h.mock_runs.length} mock`;
  historyTable($('liveRuns'), h.runs, false);
  historyTable($('mockRuns'), h.mock_runs, true);
}

function historyTable(node, runs, isMock) {
  clear(node);
  const head = ['run', 'when', 'measure', 'providers', 'arms', 'fail', ''];
  const thead = el('thead');
  const htr = el('tr');
  for (const t of head) htr.appendChild(el('th', null, t));
  thead.appendChild(htr);
  node.appendChild(thead);

  const tbody = el('tbody');
  if (!runs.length) {
    const tr = el('tr', 'empty');
    const td = el('td', null, isMock ? 'no mock runs' : 'no live runs');
    td.colSpan = head.length;
    tr.appendChild(td);
    tbody.appendChild(tr);
    node.appendChild(tbody);
    return;
  }

  for (const r of runs) {
    const tr = el('tr');

    const idTd = el('td');
    const link = el('a', 'dl', r.exec_id);
    link.href = '#';
    link.addEventListener('click', async (e) => {
      e.preventDefault();
      const doc = await (await fetch('/api/runs/' + r.exec_id)).json();
      if (doc.ok) showRun(doc.run);
    });
    if (isMock) {
      idTd.appendChild(el('span', 'badge mock', 'MOCK'));
      idTd.appendChild(document.createTextNode(' '));
    }
    idTd.appendChild(link);
    tr.appendChild(idTd);

    for (const v of [
      (r.timestamp || '').replace('T', ' ').slice(0, 19),
      r.measure || '—',
      (r.providers || []).join(','),
      String(Object.keys(r.totals || {}).length),
    ]) tr.appendChild(el('td', 'num dim', v));

    const f = el('td', 'num');
    f.textContent = String(r.failures || 0);
    if (r.failures) f.style.color = 'var(--bad)';
    tr.appendChild(f);

    const act = el('td');
    const del = el('button', 'danger', 'delete');
    del.addEventListener('click', async () => {
      if (!confirm('Delete ' + r.exec_id + '?')) return;
      del.disabled = true;
      await fetch('/api/runs/' + r.exec_id, {method: 'DELETE'});
      if (state.run && state.run.exec_id === r.exec_id) {
        $('resultsPanel').hidden = true;
        state.run = null;
      }
      loadHistory();
    });
    act.appendChild(del);
    tr.appendChild(act);

    tbody.appendChild(tr);
  }
  node.appendChild(tbody);
}

/* -------------------------------------------------------------------- boot */

$('preflightBtn').addEventListener('click', doPreflight);
$('runBtn').addEventListener('click', doRun);
$('resetBtn').addEventListener('click', () => {
  renderPicker(state.config);
  invalidatePreflight();
});
window.addEventListener('resize', () => {
  if (state.run) {
    const s = state.run.summary || {};
    drawCharts(s, s.keys || [], !!state.run.mock);
  }
});

loadConfig().then(loadHistory);
