export const VIEW_MODES = Object.freeze({
  STEP_CARD: "step-card",
  DAG: "dag",
  LOG: "log",
});

function _asArray(value) {
  return Array.isArray(value) ? value : [];
}

function _asObject(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value;
  }
  return {};
}

function _normalize_event(event) {
  return {
    eventId: String(event.eventId || ""),
    runId: String(event.runId || ""),
    threadId: String(event.threadId || ""),
    eventSeq: Number(event.eventSeq || 0),
    eventType: String(event.eventType || ""),
    issuedAt: String(event.issuedAt || ""),
    nodeId: event.nodeId || null,
    payload: _asObject(event.payload),
    checkpoint: _asObject(event.checkpoint),
    canonicalMeta: _asObject(event.canonicalMeta),
  };
}

function _normalize_history(rawHistory) {
  const history = _asObject(rawHistory);
  return {
    runId: String(history.runId || ""),
    threadId: String(history.threadId || ""),
    events: _asArray(history.events).map(_normalize_event),
    nodes: _asObject(history.nodes),
    finalState: _asObject(history.finalState),
    failureContext: history.failureContext ?? null,
    totalEvents: Number(history.totalEvents || 0),
  };
}

function _events_by_node(events) {
  const byNode = new Map();
  events.forEach((event) => {
    if (!event.nodeId) {
      return;
    }
    const nodeId = String(event.nodeId);
    const bucket = byNode.get(nodeId) || [];
    bucket.push(event);
    byNode.set(nodeId, bucket);
  });
  return byNode;
}

function _html_escape(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function _build_step_card_html(events) {
  const byNode = _events_by_node(events);
  if (byNode.size === 0) {
    return '<section class="step-card-mode"><div class="empty">No node events</div></section>';
  }

  const cards = [];
  for (const [nodeId, nodeEvents] of byNode.entries()) {
    const startSeq = nodeEvents.reduce((min, e) => (min === null || e.eventSeq < min ? e.eventSeq : min), null);
    const endSeq = nodeEvents.reduce((max, e) => (max === null || e.eventSeq > max ? e.eventSeq : max), null);
    const tokenCount = nodeEvents.filter((event) => event.eventType === "node_token").length;
    const latestType = nodeEvents[nodeEvents.length - 1]?.eventType || "none";
    cards.push(
      `<article class="step-card" data-node="${_html_escape(nodeId)}">` +
        `<header>${_html_escape(nodeId)}</header>` +
        `<div class="event-range">${startSeq}~${endSeq}</div>` +
        `<div class="latest">${_html_escape(latestType)}</div>` +
        `<div class="token-count">${tokenCount}</div>` +
      `</article>`
    );
  }
  return `<section class="step-card-mode">${cards.join("")}</section>`;
}

function _build_dag_html(events) {
  const nodes = new Set();
  const edges = [];
  let previousNode = null;
  for (const event of events) {
    if (!event.nodeId) {
      continue;
    }
    const nodeId = String(event.nodeId);
    nodes.add(nodeId);
    if (previousNode !== null && previousNode !== nodeId) {
      edges.push(`${previousNode}->${nodeId}`);
    }
    previousNode = nodeId;
  }

  const nodeMarkup = Array.from(nodes).map((nodeId) => `<li class="dag-node">${_html_escape(nodeId)}</li>`);
  const edgeMarkup = edges.map((edge) => `<li class="dag-edge">${_html_escape(edge)}</li>`);
  return `<section class="dag-mode"><ul class="nodes">${nodeMarkup.join("")}</ul><ul class="edges">${edgeMarkup.join("")}</ul></section>`;
}

function _build_log_html(events) {
  const lines = events.map((event) => {
    const nodeText = event.nodeId ? ` node=${event.nodeId}` : "";
    return `<li>${event.eventSeq} ${event.eventType}${nodeText}</li>`;
  });
  return `<section class="log-mode"><ul>${lines.join("")}</ul></section>`;
}

export function buildViewModel(rawHistory) {
  const history = _normalize_history(rawHistory);
  if (!history.runId || !history.threadId) {
    throw new TypeError("history must include runId and threadId");
  }
  const queryKey = `${history.runId}:${history.threadId}:${history.finalState?.state || "unknown"}`;
  return {
    runId: history.runId,
    threadId: history.threadId,
    queryKey,
    events: history.events,
    nodes: history.nodes,
    totalEvents: history.totalEvents || history.events.length,
    finalState: history.finalState,
    failureContext: history.failureContext,
  };
}

export function renderHistoryMode(viewModel, mode) {
  if (!Object.values(VIEW_MODES).includes(mode)) {
    throw new TypeError(`unsupported mode: ${mode}`);
  }
  const safeMode = mode;
  const events = _asArray(viewModel.events);
  let bodyHtml;
  if (safeMode === VIEW_MODES.DAG) {
    bodyHtml = _build_dag_html(events);
  } else if (safeMode === VIEW_MODES.LOG) {
    bodyHtml = _build_log_html(events);
  } else {
    bodyHtml = _build_step_card_html(events);
  }

  return `<section class="history-mode ${safeMode}" data-query="${_html_escape(viewModel.queryKey)}">${bodyHtml}</section>`;
}
