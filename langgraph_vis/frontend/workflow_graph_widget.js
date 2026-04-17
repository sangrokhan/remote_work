(function () {
  const palette = {
    planner: {
      bg: "#d7ecff",
      border: "#68a8ee",
      text: "#12365f",
    },
    executor: {
      bg: "#d7f4dd",
      border: "#7dcf90",
      text: "#1a4f2f",
    },
    refiner: {
      bg: "#fff1c7",
      border: "#e2be5e",
      text: "#5e4b17",
    },
    synthesizer: {
      bg: "#f5d9fc",
      border: "#c78ce0",
      text: "#5d2e69",
    },
    start: {
      bg: "#e7e7f8",
      border: "#9ca2df",
      text: "#32366c",
    },
    default: {
      bg: "#edf2ff",
      border: "#a9b6d7",
      text: "#2f3c56",
    },
    active: {
      bg: "#ffe8c7",
      border: "#ff9e2c",
      text: "#2e2a22",
      shadow: "0 0 0 4px rgba(255, 158, 44, 0.23)",
    },
    done: {
      border: "#3f6fe5",
      bg: "#eaf2ff",
      text: "#2749b5",
    },
  };

  const NODE_ID_ALIASES = {
    "__start__": "start",
    "__end__": "end",
  };

  function resolveNodeStyle(nodeId) {
    const lowered = String(nodeId || "").toLowerCase();
    if (NODE_ID_ALIASES[lowered]) {
      return palette[NODE_ID_ALIASES[lowered]] || palette.default;
    }
    if (palette[lowered]) {
      return palette[lowered];
    }
    return palette.default;
  }

  function parseNodeIds(rawText) {
    if (typeof rawText !== "string") {
      return [];
    }

    if (!rawText.trim().startsWith("{") || !rawText.includes("Node(")) {
      return [rawText.trim()];
    }

    const matches = [...rawText.matchAll(/'([^']+)'\s*:/g)].map((m) => m[1]);
    return matches.length > 0 ? matches : [rawText.trim()];
  }

  function normalizeNodeEntry(node) {
    if (node == null) {
      return [];
    }

    if (typeof node === "string" || typeof node === "number") {
      return parseNodeIds(String(node)).map((value) => ({ id: value }));
    }

    if (Array.isArray(node)) {
      const result = [];
      for (const item of node) {
        for (const normalized of normalizeNodeEntry(item)) {
          if (normalized.id) {
            result.push(normalized);
          }
        }
      }
      return result;
    }

    if (typeof node === "object") {
      if (typeof node.id === "string") {
        return [{ id: node.id, label: node.label || node.id }];
      }
      if (typeof node.name === "string") {
        return [{ id: node.name, label: node.label || node.name }];
      }
      if (typeof node.value === "string") {
        return [{ id: node.value, label: node.label || node.value }];
      }
      if (typeof node.key === "string") {
        return [{ id: node.key, label: node.label || node.key }];
      }
    }

    return [];
  }

  function normalizeEdgeEntry(edge) {
    if (!edge) {
      return [];
    }
    const fromRaw = edge.from || edge.source || edge.u;
    const toRaw = edge.to || edge.target || edge.v;
    if (fromRaw == null || toRaw == null) {
      return [];
    }
    const normalizedFrom = parseNodeIds(String(fromRaw).trim());
    const normalizedTo = parseNodeIds(String(toRaw).trim());
    const out = [];
    for (const source of normalizedFrom) {
      for (const target of normalizedTo) {
        out.push({
          source,
          target,
          condition: edge.condition || edge.label || null,
        });
      }
    }
    return out;
  }

  function parseWorkflowSchema(schema) {
    const nodeMap = new Map();
    const nodes = [];
    const edges = [];

    const rawNodes =
      schema?.nodes ||
      schema?.data?.nodes ||
      schema?.elements?.nodes ||
      [];
    const rawEdges =
      schema?.edges ||
      schema?.data?.edges ||
      schema?.elements?.edges ||
      [];

    for (const rawNode of rawNodes) {
      for (const entry of normalizeNodeEntry(rawNode)) {
        const id = entry.id && String(entry.id).trim();
        if (!id || nodeMap.has(id)) {
          continue;
        }
        const label = entry.label || id;
        const style = resolveNodeStyle(id);
        const normalized = {
          data: {
            id,
            label,
            bg: style.bg,
            border: style.border,
            text: style.text,
            condition: entry.condition || null,
          },
        };
        nodeMap.set(id, normalized);
        nodes.push(normalized);
      }
    }

    for (const rawEdge of rawEdges) {
      for (const edge of normalizeEdgeEntry(rawEdge)) {
        const source = String(edge.source || "").trim();
        const target = String(edge.target || "").trim();
        if (!source || !target || source === "undefined" || target === "undefined") {
          continue;
        }
        const edgeId = `${source}->${target}`;
        if (!nodeMap.has(source) || !nodeMap.has(target)) {
          if (!nodeMap.has(source)) {
            const style = resolveNodeStyle(source);
            const item = {
              data: { id: source, label: source, bg: style.bg, border: style.border, text: style.text },
            };
            nodeMap.set(source, item);
            nodes.push(item);
          }
          if (!nodeMap.has(target)) {
            const style = resolveNodeStyle(target);
            const item = {
              data: { id: target, label: target, bg: style.bg, border: style.border, text: style.text },
            };
            nodeMap.set(target, item);
            nodes.push(item);
          }
        }

        edges.push({
          data: {
            id: edgeId,
            source,
            target,
            label: edge.condition ? String(edge.condition) : "",
            width: 3,
          },
        });
      }
    }

    return { nodes, edges };
  }

  async function parseSseAndHandle(response, onNodeUpdate, onMessage, onComplete, onError) {
    if (!response.body) {
      throw new Error("응답에 스트림 본문이 없습니다.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const rawEvents = buffer.split("\n\n");
        buffer = rawEvents.pop() || "";

        for (const raw of rawEvents) {
          const event = raw.trim();
          if (!event) continue;
          let eventType = "message";
          const dataLines = [];

          for (const line of event.split("\n")) {
            if (line.startsWith("event:")) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              dataLines.push(line.slice(5).trim());
            }
          }

          const payload = dataLines.join("\n");
          if (!payload) continue;
          if (payload.startsWith("{") && payload.endsWith("}")) {
            try {
              const data = JSON.parse(payload);
              if (eventType === "done") {
                onComplete(data);
                return;
              }
              onMessage(data);
              if (typeof data?.node === "string") {
                onNodeUpdate(data.node);
              }
            } catch (err) {
              onMessage({ text: payload });
              onError(err);
            }
          } else {
            onMessage({ text: payload });
          }
        }
      }
      onComplete({ status: "complete" });
    } finally {
      reader.releaseLock();
    }
  }

  function createWorkflowGraphWidget(config) {
    const options = {
      graphEndpoint: "/graph",
      runEndpoint: "/run",
      graphContainerId: "workflow-graph",
      runButtonId: "run-btn",
      outputId: "output",
      statusId: "graph-status",
      ...config,
    };

    const graphContainer = document.getElementById(options.graphContainerId);
    const runButton = document.getElementById(options.runButtonId);
    const output = document.getElementById(options.outputId);
    const status = document.getElementById(options.statusId);

    if (!graphContainer || !runButton || !output || !status) {
      throw new Error("필수 UI 요소가 존재하지 않습니다.");
    }

    if (typeof window.cytoscape !== "function") {
      throw new Error("Cytoscape 라이브러리가 로드되지 않았습니다.");
  }

    if (typeof window.cytoscape.use === "function" && window.cytoscape.use) {
      const dagreExtension =
        window.cytoscapeDagre ||
        window.CytoscapeDagre ||
        window.dagre ||
        window.CytoscapeDagre;
      if (typeof dagreExtension === "function") {
        window.cytoscape.use(dagreExtension);
      }
    }

    const outputBuffer = [];
    const appendOutput = (message) => {
      outputBuffer.push(String(message));
      output.textContent = outputBuffer.join("\n");
      output.scrollTop = output.scrollHeight;
    };

    const updateStatus = (message) => {
      status.textContent = message || "";
    };

    const normalizeNodeId = (node) => {
      if (typeof node !== "string") {
        return String(node || "").trim();
      }
      const normalized = parseNodeIds(node);
      return normalized[0] || node;
    };

    const makeNodeClassStyle = (nodeId, active = false) => {
      if (!cy) return {};
      const target = cy.getElementById(nodeId);
      if (!target || !target.length) return {};
      const baseColor = resolveNodeStyle(nodeId);
      if (active) {
        return {
          "background-color": palette.active.bg,
          "border-color": palette.active.border,
          "border-width": 4,
        };
      }
      return {
        "background-color": baseColor.bg,
        "border-color": baseColor.border,
        color: baseColor.text,
        "border-width": 2,
      };
    };

    let cy = null;
    let isRunning = false;
    let schema = null;

    function renderStatusStyle() {
      if (!cy) return;
      cy.nodes().forEach((node) => {
        node.style(makeNodeClassStyle(node.id(), false));
      });
    }

    async function loadGraph() {
      const response = await fetch(options.graphEndpoint);
      if (!response.ok) {
        throw new Error(`Graph API 응답 오류 (${response.status})`);
      }
      const payload = await response.json();
      schema = parseWorkflowSchema(payload);
      if (!schema.nodes.length) {
        throw new Error("그래프 노드가 비어 있습니다.");
      }

      cy = window.cytoscape({
        container: graphContainer,
        elements: [...schema.nodes, ...schema.edges],
        style: [
          {
            selector: "node",
            style: {
              shape: "round-rectangle",
              content: "data(label)",
              width: 150,
              "min-width": 150,
              height: 52,
              "min-height": 52,
              padding: "8px 10px",
              "text-wrap": "wrap",
              "text-max-width": 132,
              "text-valign": "center",
              "text-halign": "center",
              "font-size": 13,
              "font-family": "system-ui, -apple-system, 'Segoe UI', Arial, sans-serif",
              "font-weight": 500,
              "corner-radius": 10,
              "border-style": "solid",
              "border-width": 2,
              "background-color": "data(bg)",
              "border-color": "data(border)",
              color: "data(text)",
              "text-opacity": 1,
              "text-background-color": "transparent",
              "text-background-opacity": 0,
            },
          },
          {
            selector: "edge",
            style: {
              width: "data(width)",
              "line-color": "#7c879d",
              "line-style": "solid",
              "curve-style": "bezier",
              "target-arrow-shape": "triangle",
              "target-arrow-color": "#7c879d",
              "arrow-scale": 1.5,
              "text-background-color": "#f9fafb",
              "font-size": 10,
              "font-family": "Arial, Segoe UI, sans-serif",
              color: "#4b5563",
            },
          },
          {
            selector: "edge[label]",
            style: {
              "text-background-opacity": 1,
              "text-background-color": "#f9fafb",
              "text-background-shape": "roundrectangle",
              "text-background-padding": "2px",
              "text-rotation": "autorotate",
              "text-margin-y": -8,
            },
          },
        ],
      });

      function applyLayout(layouts) {
        for (const candidate of layouts) {
          try {
            const layout = cy.layout(candidate);
            if (layout && typeof layout.run === "function") {
              layout.run();
              return true;
            }
            if (layout && typeof layout.start === "function") {
              layout.start();
              return true;
            }
          } catch (error) {
            continue;
          }
        }
        return false;
      }

      const hasAutoLayout = applyLayout([
        {
          name: "dagre",
          rankDir: "TB",
          nodeSep: 32,
          rankSep: 55,
          animate: false,
          spacingFactor: 1.2,
          edgeSep: 12,
          ranker: "tight-tree",
          fit: true,
          padding: 12,
        },
        {
          name: "cose",
          nodeRepulsion: 120000,
          gravity: 0.5,
          animate: false,
          fit: true,
          padding: 24,
        },
        {
          name: "breadthfirst",
          directed: true,
          spacingFactor: 1.3,
          avoidOverlap: true,
          fit: true,
          padding: 12,
        },
      ]);

      if (!hasAutoLayout) {
        const spacing = 140;
        cy.nodes().forEach((node, index) => {
          node.position({
            x: (index % 2) * spacing + 60,
            y: Math.floor(index / 2) * spacing + 60,
          });
        });
      }

      try {
        if (cy.nodes().length > 0 && typeof cy.fit === "function") {
          cy.fit(cy.elements(), 20);
        }
      } catch (error) {
        try {
          if (typeof cy.fit === "function") {
            const nodes = cy.nodes();
            const width = graphContainer.clientWidth || 320;
            const height = graphContainer.clientHeight || 240;
            const bbox = nodes.boundingBox();
            if (!width || !height || !bbox || bbox.w <= 0 || bbox.h <= 0) {
              nodes.forEach((node, index) => {
                node.position({
                  x: 60 + (index * 140) % width,
                  y: 60 + Math.floor((index * 140) / width) * 140,
                });
              });
            }
            cy.fit(nodes, 24);
          }
        } catch (_) {
          // keep default rendering as final fallback
        }
      }

      renderStatusStyle();
      const idleNodeId = getIdleActiveNodeId();
      if (idleNodeId) {
        setActiveNode(idleNodeId);
      }
      return schema;
    }

    function clearActiveState() {
      if (!cy) return;
      cy.nodes().removeClass("active");
      renderStatusStyle();
      const done = cy.nodes().filter((node) => node.data("id") === "__end__");
      done.forEach((node) => node.style(makeNodeClassStyle(node.id(), false)));
    }

    function getIdleActiveNodeId() {
      if (!schema || !schema.nodes?.length) {
        return null;
      }
      const ids = schema.nodes.map((node) => node.data?.id).filter(Boolean);
      if (ids.includes("__start__")) return "__start__";
      return ids[0] || null;
    }

    function getRunStartNodeId() {
      if (!schema || !schema.nodes?.length) {
        return null;
      }
      const ids = schema.nodes.map((node) => node.data?.id).filter(Boolean);
      if (ids.includes("planner")) return "planner";
      if (ids.includes("__start__")) return "__start__";
      return ids[0] || null;
    }

    function setActiveNode(nodeId) {
      if (!cy) return;
      const next = normalizeNodeId(nodeId);
      const el = cy.getElementById(next);
      if (!el || !el.length) {
        return;
      }
      clearActiveState();
      el.style({
        "background-color": palette.active.bg,
        "border-color": palette.active.border,
        "border-width": 4,
      });
      el.addClass("active");
    }

    async function runWorkflow() {
      if (isRunning) return;
      if (!cy) {
        await loadGraph();
      }
      isRunning = true;
      runButton.disabled = true;
      outputBuffer.length = 0;
      output.textContent = "";
      updateStatus("실행 중...");
      clearActiveState();
      const initialNodeId = getRunStartNodeId();
      if (initialNodeId) {
        setActiveNode(initialNodeId);
      }

      try {
        const response = await fetch(options.runEndpoint);
        if (!response.ok) {
          throw new Error(`Run API 응답 오류 (${response.status})`);
        }

        await parseSseAndHandle(
          response,
          (nodeId) => setActiveNode(nodeId),
          (eventData) => {
            if (eventData?.node && eventData?.state) {
              appendOutput(
                `[${eventData.node}] ${JSON.stringify(eventData.state, null, 0)}`,
              );
            } else if (typeof eventData?.text === "string") {
              appendOutput(eventData.text);
            } else {
              appendOutput(`이벤트: ${JSON.stringify(eventData, null, 0)}`);
            }
          },
          () => {
            setActiveNode("__end__");
            updateStatus("완료");
            appendOutput("실행 완료");
          },
          () => {
            updateStatus("파싱 중 일부 오류(무시)");
          },
        );
      } catch (error) {
        appendOutput(`실행 실패: ${error.message}`);
        updateStatus("실패");
      } finally {
        isRunning = false;
        runButton.disabled = false;
      }
    }

    function init() {
      runButton.addEventListener("click", runWorkflow);
      loadGraph()
        .then(() => updateStatus("그래프 로드 완료"))
        .catch((err) => {
          updateStatus("그래프 로드 실패");
          appendOutput(`그래프 로드 실패: ${err.message}`);
        });
      return {
        run: runWorkflow,
        reload: loadGraph,
      };
    }

    return {
      init,
      run: runWorkflow,
      reload: loadGraph,
      getCy: () => cy,
      getSchema: () => schema,
    };
  }

  window.createWorkflowGraphWidget = createWorkflowGraphWidget;
})();
