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
  const apiTypeField = document.getElementById("api-type-field");
  const apiTypeSelect = document.getElementById("api-type-select");
  const contextHandleField = document.getElementById("context-handle-field");
  const contextHandleSelect = document.getElementById("context-handle-select");
  const armSelect = document.getElementById("arm-select"); // hidden input, resolved arm name
  const inputModeField = document.getElementById("input-mode-field");
  const inputModeSelect = document.getElementById("input-mode-select");
  const transportField = document.getElementById("transport-field");
  const transportSelect = document.getElementById("transport");
  const algorithmSelect = document.getElementById("algorithm");
  const algorithmHint = document.getElementById("algorithm-hint");
  const form = document.getElementById("experiment-form");
  const output = document.getElementById("run-output");
  const runsTableBody = document.querySelector("#runs-table tbody");

  // Transport (TCP/QUIC) is only meaningful for the Mock card today --
  // aipt.backends.quic_mock.backend.QuicMockBackend is the only QUIC
  // Backend-protocol implementation (DESIGN.md section 7/7.1). Hidden
  // entirely for every other card, and forced back to "http1" the moment
  // a non-mock card is selected, so a stray "http3" from a previous Mock
  // selection can never silently ride along into a public_ai/local_llm
  // request that has no idea what to do with it.
  function applyTransportAvailability(key) {
    const transportAllowed = key === "mock";
    if (!transportAllowed && transportSelect.value !== "http1") {
      transportSelect.value = "http1";
    }
    transportField.style.display = transportAllowed ? "block" : "none";
    populateAlgorithmOptions(transportSelect.value);
  }

  // Swaps the Congestion algorithm dropdown's option list between the
  // kernel's TCP modules (config.congestion_algorithms) and aioquic's
  // registered QUIC algorithms (config.quic_congestion_algorithms) -- the
  // two are never the same namespace (a kernel module name like "bbr" and
  // a QUIC congestion-control name like "idle_probe" mean different
  // things to different code paths server-side), so the option list itself
  // must change, not just which one gets submitted.
  function populateAlgorithmOptions(transport) {
    const isQuic = transport === "http3";
    const names = isQuic
      ? CONFIG.quic_congestion_algorithms || []
      : CONFIG.congestion_algorithms || [];
    const reason = isQuic
      ? CONFIG.quic_congestion_algorithms_reason
      : CONFIG.congestion_algorithms_reason;

    algorithmSelect.innerHTML = "";
    const defaultOpt = document.createElement("option");
    defaultOpt.value = "";
    defaultOpt.textContent = isQuic ? "(reno, aioquic default)" : "(kernel default)";
    algorithmSelect.appendChild(defaultOpt);
    for (const name of names) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      algorithmSelect.appendChild(opt);
    }
    algorithmHint.textContent = names.length ? "" : reason || "";
  }

  transportSelect.addEventListener("change", () => populateAlgorithmOptions(transportSelect.value));

  // Every backend card now resolves its arm through two visible pickers
  // instead of one flat arm dropdown: API Type first (the actual billable
  // HTTP endpoint -- Gemini's generateContent vs Interaction API, ChatGPT's
  // Chat Completion vs Responses API, ...), then Context Handle (how that
  // API's conversation history is carried -- Default/No Context/Force
  // Caching under generateContent; Default/Inline/Stateless under
  // Interaction or Responses; a single Default under APIs with only one
  // calling convention). Each (api_type, context_handle) pair maps to
  // exactly one backend-validated arm name, which lands in the hidden
  // #arm-select input the submit handler already reads.
  function populateArms(key) {
    const card = BACKENDS_BY_KEY[key] || {};
    const apiTypes = card.api_types || [];
    const hasApiTypes = apiTypes.length > 0;
    apiTypeField.style.display = hasApiTypes ? "block" : "none";
    contextHandleField.style.display = hasApiTypes ? "block" : "none";

    if (!hasApiTypes) {
      armSelect.value = "";
      return;
    }

    apiTypeSelect.innerHTML = "";
    for (const group of apiTypes) {
      const opt = document.createElement("option");
      opt.value = group.key;
      opt.textContent = group.label;
      apiTypeSelect.appendChild(opt);
    }
    populateContextHandles(key, apiTypeSelect.value);
  }

  function populateContextHandles(key, apiTypeKey) {
    const card = BACKENDS_BY_KEY[key] || {};
    const group = (card.api_types || []).find((g) => g.key === apiTypeKey);
    const handles = group ? group.context_handles : [];
    contextHandleSelect.innerHTML = "";
    for (const handle of handles) {
      const opt = document.createElement("option");
      opt.value = handle.key;
      opt.textContent = handle.label;
      opt.dataset.arm = handle.arm;
      contextHandleSelect.appendChild(opt);
    }
    resolveArm();
  }

  function resolveArm() {
    const opt = contextHandleSelect.options[contextHandleSelect.selectedIndex];
    armSelect.value = opt ? opt.dataset.arm : "";
  }

  apiTypeSelect.addEventListener("change", () => {
    populateContextHandles(backendSelect.value, apiTypeSelect.value);
  });
  contextHandleSelect.addEventListener("change", resolveArm);

  function toggleBackendFields(key) {
    document.querySelectorAll(".backend-fields").forEach((el) => {
      el.style.display = el.dataset.key === key ? "block" : "none";
    });
  }

  // "dummy" input mode is mock-only (byte-size filler, no real content --
  // meaningless for a backend that talks to a real model/engine). Public
  // AI/Local LLM never show it: force "record" and hide the mode switch
  // itself rather than leaving a dead option in the dropdown.
  function applyInputModeAvailability(key) {
    const dummyAllowed = key === "mock";
    if (!dummyAllowed && inputModeSelect.value === "dummy") {
      inputModeSelect.value = "record";
    }
    inputModeField.style.display = dummyAllowed ? "block" : "none";
    toggleInputModeFields(dummyAllowed ? inputModeSelect.value : "record");
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
    applyTransportAvailability(key);
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
    const inputMode = key === "mock" ? (fd.get("input_mode") || "record") : "record";

    const body = {
      backend,
      engine,
      arm: armSelect.value,
      model,
      system: "", // resolved server-side from the record or left blank (dummy)
      measure: fd.get("measure") || "bytes",
      input_mode: inputMode,
      record_id: fd.get("record_id") || "",
      // Fallbacks mirror the form's own dummy-fields defaults (operator
      // spec) and RunRequest's in routes_run.py, so a submit with an
      // emptied/missing field still lands on the same stress-shaped
      // defaults rather than an inconsistent zero.
      system_prompt_bytes: Number(fd.get("system_prompt_bytes") || 20000),
      turn_user_msg_bytes: Number(fd.get("turn_user_msg_bytes") || 1000),
      num_turns: Number(fd.get("num_turns") || 10),
      mock_response_bytes: Number(fd.get("mock_response_bytes") || 1000),
      inference_delay_ms: Number(fd.get("inference_delay_ms") || 1000),
      algorithm: fd.get("algorithm") || null,
      // "http1" (kernel TCP, default) or "http3" (QUIC, mock-only spike --
      // see routes_run.RunRequest.transport's docstring). The dropdown is
      // hidden for every non-mock card (applyTransportAvailability()
      // above forces it back to "http1" the instant a different card is
      // selected), so this always reflects a value the currently-selected
      // backend can actually honour.
      transport: fd.get("transport") || "http1",
      // Checkbox default is checked in the HTML (operator decision,
      // 2026-08-27: capture on by default) -- FormData omits an unchecked
      // checkbox entirely, so its *presence* is the true/false signal,
      // not a value comparison.
      capture: fd.has("capture"),
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
