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
  const algorithmSelect = document.getElementById("algorithm");
  const algorithmHint = document.getElementById("algorithm-hint");
  const form = document.getElementById("experiment-form");
  const output = document.getElementById("run-output");
  const runsTableBody = document.querySelector("#runs-table tbody");
  // Referenced by updateParamSummary(), which selectBackend() calls during
  // page-load initialization (near the bottom of this file) -- must be a
  // top-level const alongside the other DOM refs so it's already
  // initialized by the time that first call happens (a const declared
  // further down is in the temporal dead zone until its own line runs,
  // which caused a ReferenceError here when this lived next to
  // updateParamSummary() instead).
  const paramSummaryContent = document.getElementById("param-summary-content");

  // Network Gateway profile (DESIGN.md 4.7 B11) and idle-reset toggle
  // (2026-09-01 ooo interview) are both standing state on a backend
  // container, not a RunRequest field -- only meaningful for mock/local_llm
  // (the containers this project's Gateway/idle-reset admin actually
  // reach; public_ai is the real internet, and has no admin route). Hidden
  // entirely for every other card, same pattern as toggleBackendFields.
  const gatewayProfileField = document.getElementById("gateway-profile-field");
  const gatewayProfileSelect = document.getElementById("gateway-profile-select");
  const gatewayProfileApply = document.getElementById("gateway-profile-apply");
  const gatewayProfileStatus = document.getElementById("gateway-profile-status");
  const idleResetField = document.getElementById("idle-reset-field");
  const idleResetSelect = document.getElementById("idle-reset-select");
  const idleResetApply = document.getElementById("idle-reset-apply");
  const idleResetStatus = document.getElementById("idle-reset-status");

  // idle-reset ALWAYS targets `web` itself (see routes_gateway.py's
  // 2026-09-02 redesign docstring): it's the CLIENT's (this `web`
  // container's) own send-side cwnd that slow-start-after-idle resets for
  // the metric that matters (next-turn request upload latency), and that
  // is true no matter which backend card is selected -- unlike Gateway
  // profile (which only means something for traffic that actually
  // crosses the `gateway` netem hop, i.e. mock/local_llm), so the
  // idle-reset field/Apply is never hidden and never keyed by backend.
  // /api/idle-reset takes no backend query param at all any more (the
  // mock/local_llm admin-proxy path was removed 2026-09-02, dead code
  // after the redesign above).

  function applyGatewayIdleResetAvailability(key) {
    // Gateway sits between web and {mock-server, local-llm} -- both
    // backends' traffic crosses it (DESIGN.md 4.7), so the profile
    // control is relevant for either, but not public_ai (real internet).
    const gatewayAllowed = key === "mock" || key === "local_llm";
    gatewayProfileField.style.display = gatewayAllowed ? "block" : "none";
  }

  async function refreshIdleResetStatus() {
    idleResetStatus.textContent = "checking...";
    try {
      const res = await fetch(`/api/idle-reset`);
      const payload = await res.json();
      if (payload.ok && payload.enabled !== null) {
        idleResetSelect.value = payload.enabled ? "1" : "0";
        idleResetStatus.textContent = `current: ${payload.enabled ? "enabled" : "disabled"} (${payload.reason})`;
      } else {
        idleResetStatus.textContent = `unavailable: ${payload.reason || "unknown"}`;
      }
    } catch (e) {
      idleResetStatus.textContent = `check failed: ${String(e)}`;
    }
    updateParamSummary();
  }

  gatewayProfileApply.addEventListener("click", async () => {
    gatewayProfileStatus.textContent = "applying...";
    try {
      const res = await fetch(`/api/gateway/profile?profile=${encodeURIComponent(gatewayProfileSelect.value)}`, {
        method: "POST",
      });
      const payload = await res.json();
      gatewayProfileStatus.textContent = payload.ok
        ? `applied: ${gatewayProfileSelect.value}`
        : `failed: ${payload.reason || JSON.stringify(payload)}`;
    } catch (e) {
      gatewayProfileStatus.textContent = `request failed: ${String(e)}`;
    }
    updateParamSummary();
  });

  async function applyIdleReset(enabledValue) {
    idleResetStatus.textContent = "applying...";
    try {
      const res = await fetch(
        `/api/idle-reset?enabled=${enabledValue === "1"}`,
        { method: "POST" }
      );
      const payload = await res.json();
      idleResetStatus.textContent = payload.write_ok
        ? `applied: ${payload.enabled ? "enabled" : "disabled"}`
        : `failed: ${payload.write_reason || payload.reason || JSON.stringify(payload)}`;
      return payload;
    } catch (e) {
      idleResetStatus.textContent = `request failed: ${String(e)}`;
      return null;
    } finally {
      updateParamSummary();
    }
  }

  idleResetApply.addEventListener("click", () => applyIdleReset(idleResetSelect.value));

  // -- Reset-to-defaults after a run (operator request, 2026-09-02) -------
  //
  // These two controls are standing state, not per-run parameters (see
  // both fields' comments in _experiment_form.html), so a run leaves them
  // exactly as they were even after the run that needed them is over. To
  // stop a forgotten "disabled"/"wireless" setting from silently biasing
  // the *next* operator's next run, every completed /api/run POST (success
  // or failure -- see the finally in the submit handler below) re-applies
  // the documented defaults: idle-reset -> enabled (Linux default,
  // idle-reset-select's own `selected` option) and Gateway profile ->
  // custom (GATEWAY_DELAY_MS-based, gateway-profile-select's own
  // `selected` option). This actually re-POSTs to the two APIs -- it is
  // not just an HTML dropdown reset -- so the container/kernel state is
  // truly back to baseline, not just the UI's display of it.
  const IDLE_RESET_DEFAULT = "1"; // matches the <option selected> in the form
  const GATEWAY_PROFILE_DEFAULT = "custom"; // matches the <option selected> in the form

  async function resetStandingStateToDefaults() {
    gatewayProfileSelect.value = GATEWAY_PROFILE_DEFAULT;
    idleResetSelect.value = IDLE_RESET_DEFAULT;
    await Promise.all([
      (async () => {
        gatewayProfileStatus.textContent = "resetting to default...";
        try {
          const res = await fetch(`/api/gateway/profile?profile=${GATEWAY_PROFILE_DEFAULT}`, { method: "POST" });
          const payload = await res.json();
          gatewayProfileStatus.textContent = payload.ok
            ? `reset to default: ${GATEWAY_PROFILE_DEFAULT}`
            : `reset failed: ${payload.reason || JSON.stringify(payload)}`;
        } catch (e) {
          gatewayProfileStatus.textContent = `reset request failed: ${String(e)}`;
        }
      })(),
      applyIdleReset(IDLE_RESET_DEFAULT),
    ]);
    updateParamSummary();
  }

  // Congestion algorithm dropdown's option list -- the kernel's actually
  // loaded TCP modules (config.congestion_algorithms), never a fixed
  // guess that could offer a name this box has no module loaded for.
  function populateAlgorithmOptions() {
    const names = CONFIG.congestion_algorithms || [];
    const reason = CONFIG.congestion_algorithms_reason;

    algorithmSelect.innerHTML = "";
    const defaultOpt = document.createElement("option");
    defaultOpt.value = "";
    defaultOpt.textContent = "(kernel default)";
    algorithmSelect.appendChild(defaultOpt);
    for (const name of names) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      algorithmSelect.appendChild(opt);
    }
    algorithmHint.textContent = names.length ? "" : reason || "";
  }

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
    applyGatewayIdleResetAvailability(key);
    populateAlgorithmOptions();
    updateParamSummary();
  }

  document.querySelectorAll(".select-backend").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      selectBackend(btn.dataset.key);
      document.querySelectorAll(".card").forEach((c) => c.classList.remove("selected"));
      btn.closest(".card").classList.add("selected");
    });
  });

  inputModeSelect.addEventListener("change", () => {
    toggleInputModeFields(inputModeSelect.value);
    updateParamSummary();
  });

  // Any other form field changing (measure, capture, algorithm,
  // input mode, record/dummy fields, per-backend model fields, cache
  // options...) should also refresh the summary -- rather than wiring a
  // listener per field, delegate on the form itself for both "change"
  // (selects/checkboxes) and "input" (free-typed text/number fields).
  form.addEventListener("change", updateParamSummary);
  form.addEventListener("input", updateParamSummary);

  // Initial state: first implemented card (Gemini, if it's ready to run).
  const firstImplemented = (CONFIG.ui_backends || []).find((b) => b.implemented);
  if (firstImplemented) selectBackend(firstImplemented.key);

  // idle-reset always targets web_client regardless of backend selection
  // (see IDLE_RESET_BACKEND above), so its status is fetched once at
  // load time rather than re-fetched on every backend switch.
  refreshIdleResetStatus();

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
          <td>${t.cache_bytes_saved || 0}</td>
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
        <thead><tr><th>turn</th><th>wire_sent</th><th>wire_recv</th><th>ttlt_ms</th><th>goodput_bps</th><th>cache_bytes_saved</th><th>error</th></tr></thead>
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

  // Builds the exact JSON body the next /api/run POST would carry, given
  // the form's CURRENT field values -- shared by the submit handler and
  // updateParamSummary() so the pre-run summary can never drift from what
  // actually gets sent (single source of truth, not two hand-maintained
  // copies of the same field list).
  function buildRunBody() {
    const fd = new FormData(form);
    const key = backendSelect.value;
    const card = BACKENDS_BY_KEY[key] || {};
    const backend = card.backend || key;
    const engine = card.engine || null;
    const model = fd.get(`${key}_model`) || "";
    const inputMode = key === "mock" ? (fd.get("input_mode") || "record") : "record";

    return {
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
      // Checkbox default is checked in the HTML (operator decision,
      // 2026-08-27: capture on by default) -- FormData omits an unchecked
      // checkbox entirely, so its *presence* is the true/false signal,
      // not a value comparison.
      capture: fd.has("capture"),
      // local_llm-only request cache (docs/engine_gateway_caching_seed.md).
      // Default UNCHECKED in the HTML (opt-in, unlike capture above), so
      // fd.has() correctly reads "off" when the local_llm card wasn't even
      // showing this fieldset (a non-local_llm run never has this field
      // checked, and the backend ignores these fields for other backends
      // anyway -- see routes_run._build_backend()).
      cache_enabled: fd.has("cache_enabled"),
      cache_threshold_bytes: Number(fd.get("cache_threshold_bytes") || 200),
    };
  }

  // Parameter summary panel (operator request, 2026-09-02): shows every
  // value the next /api/run POST would carry (from buildRunBody(), so it
  // can never fall out of sync with what actually gets sent) plus the two
  // standing-state controls' *applied* status lines (idleResetStatus.
  // textContent / gatewayProfileStatus.textContent), which reflect the
  // real container/kernel state as last confirmed by the backend, not
  // merely the dropdown's current selection. paramSummaryContent itself is
  // declared at the top of this file, next to the other DOM refs (see
  // that declaration's comment for why).
  function updateParamSummary() {
    if (!paramSummaryContent) return;
    let body;
    try {
      body = buildRunBody();
    } catch (e) {
      paramSummaryContent.textContent = `(unable to build summary: ${String(e)})`;
      return;
    }
    const lines = [
      `gateway_profile (applied): ${gatewayProfileStatus.textContent || "(unknown -- not yet checked)"}`,
      `idle_reset (applied, client-side): ${idleResetStatus.textContent || "(unknown -- not yet checked)"}`,
      "--- next run body ---",
      JSON.stringify(body, null, 2),
    ];
    paramSummaryContent.textContent = lines.join("\n");
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const body = buildRunBody();

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
    } finally {
      // Standing state (Gateway profile / idle-reset) must not silently
      // carry a non-default setting into whatever the operator runs next
      // -- see resetStandingStateToDefaults()'s comment above.
      await resetStandingStateToDefaults();
    }
    refreshRuns();
  });

  refreshRuns();
  updateParamSummary();
})();
