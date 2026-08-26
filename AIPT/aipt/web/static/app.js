// aipt/web/static/app.js -- backend-selection toggle, fixed/dummy input
// mode toggle, /api/run driver, minimal result/recent-runs rendering. No
// chart library (DESIGN.md 5: "차트는 이번 범위 밖으로 남겨도 됨") -- text/
// table output only.

(function () {
  "use strict";

  const CONFIG = window.__AIPT_CONFIG__ || { ui_backends: [] };
  const BACKENDS_BY_KEY = Object.fromEntries(
    (CONFIG.ui_backends || []).map((b) => [b.key, b])
  );

  const backendSelect = document.getElementById("backend-select"); // hidden input
  const armSelect = document.getElementById("arm-select");
  const inputModeField = document.getElementById("input-mode-field");
  const inputModeSelect = document.getElementById("input-mode-select");
  const form = document.getElementById("experiment-form");
  const output = document.getElementById("run-output");
  const runsTableBody = document.querySelector("#runs-table tbody");

  function populateArms(key) {
    const arms = (BACKENDS_BY_KEY[key] || {}).arms || [];
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

  function toggleBackendFields(key) {
    document.querySelectorAll(".backend-fields").forEach((el) => {
      el.style.display = el.dataset.key === key ? "block" : "none";
    });
  }

  // "dummy" input mode is mock-only (byte-size filler, no real content --
  // meaningless for a backend that talks to a real model/engine). Public
  // AI/Local LLM never show it: force "fixed" and hide the mode switch
  // itself rather than leaving a dead option in the dropdown.
  function applyInputModeAvailability(key) {
    const dummyAllowed = key === "mock";
    if (!dummyAllowed && inputModeSelect.value === "dummy") {
      inputModeSelect.value = "fixed";
    }
    inputModeField.style.display = dummyAllowed ? "block" : "none";
    toggleInputModeFields(dummyAllowed ? inputModeSelect.value : "fixed");
  }

  function toggleInputModeFields(mode) {
    document.querySelectorAll(".input-mode-fields").forEach((el) => {
      el.style.display = el.dataset.mode === mode ? "block" : "none";
    });
  }

  function selectBackend(key) {
    backendSelect.value = key;
    populateArms(key);
    toggleBackendFields(key);
    applyInputModeAvailability(key);
  }

  document.querySelectorAll(".select-backend").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      selectBackend(btn.dataset.key);
      document.querySelectorAll(".card").forEach((c) => c.classList.remove("selected"));
      btn.closest(".card").classList.add("selected");
    });
  });

  inputModeSelect.addEventListener("change", () => toggleInputModeFields(inputModeSelect.value));

  // Initial state: first implemented card (Gemini, if it's ready to run).
  const firstImplemented = (CONFIG.ui_backends || []).find((b) => b.implemented);
  if (firstImplemented) selectBackend(firstImplemented.key);

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
    const key = backendSelect.value;
    const card = BACKENDS_BY_KEY[key] || {};
    const backend = card.backend || key;
    const engine = card.engine || null;
    // Model field is namespaced per engine/backend (gemini_model,
    // openai_model, local_llm_model, ...) so each card's fieldset can
    // carry its own placeholder/default without one shared "model" field
    // silently applying the wrong vendor's model name.
    const model = fd.get(`${key}_model`) || "";
    const inputMode = key === "mock" ? (fd.get("input_mode") || "fixed") : "fixed";

    const body = {
      backend,
      engine,
      arm: armSelect.value,
      model,
      system: "", // resolved server-side from the fixture (fixed) or left blank (dummy)
      measure: fd.get("measure") || "bytes",
      input_mode: inputMode,
      fixture_name: fd.get("fixture_name") || "",
      system_prompt_bytes: Number(fd.get("system_prompt_bytes") || 0),
      turn_user_msg_bytes: Number(fd.get("turn_user_msg_bytes") || 0),
      num_turns: Number(fd.get("num_turns") || 3),
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
