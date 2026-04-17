import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

const INDEX_HTML_PATH = join(process.cwd(), "frontend", "index.html");
const APP_JS_PATH = join(process.cwd(), "frontend", "app.js");
const STYLE_PATH = join(process.cwd(), "frontend", "styles.css");

function readIndexHtml() {
  return readFileSync(INDEX_HTML_PATH, "utf8");
}

test("frontend page is a run-only demo UI", () => {
  const html = readIndexHtml();
  const appJs = readFileSync(APP_JS_PATH, "utf8");
  const styleCss = readFileSync(STYLE_PATH, "utf8");

  assert.ok(html.includes("id=\"run-btn\""), "run button must exist");
  assert.ok(html.includes(">Run</button>"), "run button label should be Run");
  assert.ok(html.includes('href="./styles.css"'), "styles.css should be loaded");
  assert.ok(html.includes('src="./app.js"'), "app.js should be loaded");
  assert.ok(styleCss.includes(".node-card"), "styles.css should contain node card styling");
  assert.ok(appJs.includes("startRun"), "startRun handler should exist");
  assert.ok(appJs.includes("api/workflows/langgraph_demo_workflow_v1/schema"), "schema API should be used");
  assert.ok(appJs.includes("/api/runs"), "run API should be used");
  assert.ok(appJs.includes("/demo/run"), "demo run API should be used");
  assert.ok(appJs.includes("/events/stream"), "event stream endpoint should be used");
  assert.ok(appJs.includes("node_started"), "node_started event handling should exist");
  assert.ok(appJs.includes("node_completed"), "node_completed event handling should exist");
  assert.ok(appJs.includes("run_completed"), "run_completed event handling should exist");
});

test("frontend page should not include legacy demo controls", () => {
  const html = readIndexHtml();

  assert.equal(html.includes('id="status"'), false, "legacy status badge should not exist");
  assert.equal(html.includes("run-meta"), false, "run meta text block should not exist");
  assert.equal(html.includes("event-stream"), false, "event log panel should not exist");
});
