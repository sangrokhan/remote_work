const runButton = document.getElementById("run-btn");
const graphOutput = document.getElementById("graph-output");
const output = document.getElementById("output");

function appendLine(message) {
  output.textContent += `${message}\n`;
}

function formatGraphJson(data) {
  return JSON.stringify(data, null, 2);
}

async function loadGraph() {
  graphOutput.textContent = "그래프 로딩 중...";
  try {
    const response = await fetch("/graph");
    if (!response.ok) {
      graphOutput.textContent = `그래프 로딩 실패: HTTP ${response.status}`;
      return;
    }

    const data = await response.json();
    graphOutput.textContent = `Graph Schema\n${formatGraphJson(data)}`;
  } catch (error) {
    graphOutput.textContent = `그래프 로딩 실패: ${error.message}`;
  }
}

runButton.addEventListener("click", async () => {
  output.textContent = "";
  runButton.disabled = true;
  appendLine("실행 시작");

  const response = await fetch("/run");
  if (!response.ok) {
    appendLine(`실행 실패: HTTP ${response.status}`);
    runButton.disabled = false;
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop();

    for (const raw of events) {
      if (!raw.startsWith("data:")) {
        continue;
      }
      const payload = raw.slice(5).trim();
      try {
        const parsed = JSON.parse(payload);
        appendLine(
          `[${parsed.node}] ${JSON.stringify(parsed.state, null, 0)}`,
        );
      } catch {
        appendLine(payload);
      }
    }
  }

  if (buffer && buffer.startsWith("data:")) {
    try {
      const parsed = JSON.parse(buffer.slice(5).trim());
      appendLine(`[${parsed.node}] ${JSON.stringify(parsed.state, null, 0)}`);
    } catch {
      appendLine(buffer.trim());
    }
  }

  appendLine("실행 완료");
  runButton.disabled = false;
});

loadGraph();
