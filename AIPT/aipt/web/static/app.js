// aipt/web/static/app.js -- backend-selection toggle, /api/run driver,
// minimal result/recent-runs rendering. No chart library (DESIGN.md 5:
// "차트는 이번 범위 밖으로 남겨도 됨") -- text/table output only.

(function () {
  "use strict";

  const CONFIG = window.__AIPT_CONFIG__ || { backends: [] };
  const BACKENDS_BY_NAME = Object.fromEntries(
    (CONFIG.backends || []).map((b) => [b.name, b])
  );

  const backendSelect = document.getElementById("backend-select");
  const armSelect = document.getElementById("arm-select");
  const form = document.getElementById("experiment-form");
  const output = document.getElementById("run-output");
  const runsTableBody = document.querySelector("#runs-table tbody");

  function populateArms(backendName) {
    const arms = (BACKENDS_BY_NAME[backendName] || {}).arms || [];
    armSelect.innerHTML = "";
    for (const arm of arms) {
      const opt = document.createElement("option");
      opt.value = arm;
      opt.textContent = arm;
      armSelect.appendChild(opt);
    }
    if (arms.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(no arms -- backend not implemented yet)";
      armSelect.appendChild(opt);
    }
  }

  function toggleBackendFields(backendName) {
    document.querySelectorAll(".backend-fields").forEach((el) => {
      el.style.display = el.dataset.backend === backendName ? "block" : "none";
    });
  }

  function selectBackend(name) {
    backendSelect.value = name;
    populateArms(name);
    toggleBackendFields(name);
  }

  backendSelect.addEventListener("change", () => selectBackend(backendSelect.value));

  document.querySelectorAll(".select-backend").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      selectBackend(btn.dataset.backend);
    });
  });

  // Initial state: first implemented backend.
  const firstImplemented = (CONFIG.backends || []).find((b) => b.implemented);
  if (firstImplemented) selectBackend(firstImplemented.name);

  function renderResult(payload) {
    if (!payload.ok) {
      output.innerHTML = `<p class="status warn">Error: ${escapeHtml(
        payload.error || "unknown error"
      )}</p>`;
      return;
    }
    const run = payload.run;
    const rows = (run.turns || [])
      .map(
        (t) => `<tr>
          <td>${t.turn}</td>
          <td>${t.wire_sent}</td>
          <td>${t.wire_recv}</td>
          <td>${t.ttlt_ms}</td>
          <td>${t.goodput_bps}</td>
          <td>${escapeHtml(t.error || "")}</td>
        </tr>`
      )
      .join("");
    output.innerHTML = `
      <p><strong>exec_id:</strong> ${run.exec_id} &middot;
         <strong>backend:</strong> ${run.backend} &middot;
         <strong>arm:</strong> ${run.arm} &middot;
         <strong>elapsed:</strong> ${run.elapsed_s}s</p>
      <table class="turns-table">
        <thead><tr><th>turn</th><th>wire_sent</th><th>wire_recv</th><th>ttlt_ms</th><th>goodput_bps</th><th>error</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="downloads">
        <a href="/api/runs/${run.exec_id}/turns.csv">turns.csv</a> ·
        <a href="/api/runs/${run.exec_id}/cwnd.csv">cwnd.csv</a> ·
        <a href="/api/runs/${run.exec_id}/bundle.zip">bundle.zip</a>
      </p>`;
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  async function refreshRuns() {
    try {
      const res = await fetch("/api/runs");
      const runs = await res.json();
      runsTableBody.innerHTML = runs
        .map(
          (r) => `<tr>
            <td>${r.exec_id}</td><td>${r.backend}</td><td>${r.arm}</td>
            <td>${r.turn_count}</td><td>${r.mock}</td><td>${escapeHtml(r.error || "")}</td>
          </tr>`
        )
        .join("");
    } catch (e) {
      // best-effort; the run itself already succeeded/failed independently
    }
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(form);
    const backend = fd.get("backend");
    const body = {
      backend,
      arm: armSelect.value,
      model: fd.get("model") || "",
      system: fd.get("system") || "",
      turns: (fd.get("turns") || "")
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean),
      measure: fd.get("measure") || "bytes",
      mock_response_bytes: Number(fd.get("mock_response_bytes") || 400),
      inference_delay_ms: Number(fd.get("inference_delay_ms") || 0),
      algorithm: fd.get("algorithm") || null,
    };

    output.textContent = "Running...";
    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await res.json();
      renderResult(payload);
    } catch (e) {
      output.innerHTML = `<p class="status warn">Request failed: ${escapeHtml(String(e))}</p>`;
    }
    refreshRuns();
  });

  refreshRuns();
})();
