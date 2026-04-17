const API_ORIGIN = window.location.origin;
const WORKFLOW_ID = "langgraph_demo_workflow_v1";

const runButton = document.getElementById("run-btn");
const workflowContainer = document.getElementById("workflow");
const nodeCardById = new Map();
const orderedNodeIds = [];
let eventSource = null;
let isRunning = false;

function setNodeState(nodeId, nextState) {
  const card = nodeCardById.get(nodeId);
  if (!card) {
    return;
  }

  card.classList.remove("ready", "running", "done");
  if (nextState) {
    card.classList.add(nextState);
  }
}

function resetNodeStates() {
  for (const nodeId of orderedNodeIds) {
    setNodeState(nodeId);
  }
  const startNodeId = orderedNodeIds[0];
  if (startNodeId) {
    setNodeState(startNodeId, "ready");
  }
}

function renderWorkflow(schema) {
  workflowContainer.innerHTML = "";
  nodeCardById.clear();
  orderedNodeIds.length = 0;

  const nodes = Array.isArray(schema.nodes)
    ? [...schema.nodes].sort((a, b) => Number(a.order || 0) - Number(b.order || 0))
    : [];

  if (nodes.length === 0) {
    workflowContainer.innerHTML = "<div class=\"empty\">노드를 찾을 수 없습니다.</div>";
    return;
  }

  nodes.forEach((node, index) => {
    const nodeId = String(node.id);
    orderedNodeIds.push(nodeId);

    const card = document.createElement("article");
    card.className = "node-card";
    card.dataset.nodeId = nodeId;

    const label = document.createElement("h2");
    label.className = "node-label";
    label.textContent = `${index + 1}. ${node.label}`;

    const meta = document.createElement("p");
    meta.className = "node-meta";
    meta.textContent = node.description || "";

    card.appendChild(label);
    card.appendChild(meta);
    workflowContainer.appendChild(card);

    if (index + 1 < nodes.length) {
      const arrow = document.createElement("div");
      arrow.className = "arrow";
      arrow.textContent = "↓";
      workflowContainer.appendChild(arrow);
    }

    nodeCardById.set(nodeId, card);
  });

  resetNodeStates();
}

function stopStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

function applyEvent(event) {
  const eventNodeId = event?.payload?.nodeId || event?.nodeId;
  if (event.eventType === "node_started" && eventNodeId) {
    for (const nodeId of orderedNodeIds) {
      if (nodeCardById.get(nodeId).classList.contains("running")) {
        setNodeState(nodeId, "done");
      }
    }
    setNodeState(String(eventNodeId), "running");
    return;
  }

  if (event.eventType === "node_completed" && eventNodeId) {
    setNodeState(String(eventNodeId), "done");
    return;
  }

  if (event.eventType === "run_completed") {
    isRunning = false;
    runButton.disabled = false;
    runButton.textContent = "Run";
    stopStream();
  }
}

function subscribeEvents(runId) {
  stopStream();
  eventSource = new EventSource(`${API_ORIGIN}/api/runs/${encodeURIComponent(runId)}/events/stream`);
  eventSource.addEventListener("run-event", (event) => {
    applyEvent(JSON.parse(event.data));
  });
  eventSource.addEventListener("error", () => {
    if (isRunning) {
      stopStream();
    }
  });
}

async function startRun() {
  if (isRunning) {
    return;
  }

  if (runButton.disabled) {
    return;
  }

  try {
    isRunning = true;
    runButton.disabled = true;
    runButton.textContent = "실행 중…";
    resetNodeStates();

    const runId = `run_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
    const threadId = `thread_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

    const createResponse = await fetch(`${API_ORIGIN}/api/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        runId,
        threadId,
        workflowId: WORKFLOW_ID,
      }),
    });
    if (!createResponse.ok) {
      const message = await createResponse.text();
      throw new Error(`run 생성 실패: ${createResponse.status} ${message}`);
    }

    subscribeEvents(runId);

    const runResponse = await fetch(`${API_ORIGIN}/api/runs/${encodeURIComponent(runId)}/demo/run`, {
      method: "POST",
    });
    if (!runResponse.ok) {
      const message = await runResponse.text();
      throw new Error(`demo 실행 실패: ${runResponse.status} ${message}`);
    }
  } catch (error) {
    isRunning = false;
    runButton.disabled = false;
    runButton.textContent = "Run";
    stopStream();
    console.error(error);
    alert(error.message || "실행 중 오류가 발생했습니다.");
  }
}

async function init() {
  const response = await fetch(`${API_ORIGIN}/api/workflows/${WORKFLOW_ID}/schema`);
  if (!response.ok) {
    throw new Error(`workflow 로드 실패: ${response.status}`);
  }
  const schema = await response.json();
  renderWorkflow(schema);
}

runButton.addEventListener("click", () => {
  startRun();
});

init().catch((error) => {
  workflowContainer.innerHTML = "<div class=\"empty\">workflow를 불러오지 못했습니다.</div>";
  runButton.disabled = true;
  alert(error.message || "초기화 실패");
});
