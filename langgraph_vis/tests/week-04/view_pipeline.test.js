import test from "node:test";
import assert from "node:assert/strict";

import {
  VIEW_MODES,
  buildViewModel,
  renderHistoryMode,
} from "./view_pipeline.js";

const sampleHistory = {
  runId: "run_1",
  threadId: "thread_1",
  events: [
    {
      eventId: "evt-001",
      runId: "run_1",
      threadId: "thread_1",
      eventSeq: 1,
      eventType: "run_started",
      issuedAt: "2026-04-14T00:00:00Z",
      nodeId: null,
      payload: {},
      checkpoint: { state: "running", eventSeq: 1 },
    },
    {
      eventId: "evt-002",
      runId: "run_1",
      threadId: "thread_1",
      eventSeq: 2,
      eventType: "node_started",
      issuedAt: "2026-04-14T00:00:01Z",
      nodeId: "intent_parser",
      payload: { nodeId: "intent_parser" },
      checkpoint: { state: "running", eventSeq: 2 },
    },
    {
      eventId: "evt-003",
      runId: "run_1",
      threadId: "thread_1",
      eventSeq: 3,
      eventType: "node_token",
      issuedAt: "2026-04-14T00:00:02Z",
      nodeId: "intent_parser",
      payload: { nodeId: "intent_parser", token: "안녕" },
      checkpoint: { state: "running", eventSeq: 3 },
    },
    {
      eventId: "evt-004",
      runId: "run_1",
      threadId: "thread_1",
      eventSeq: 4,
      eventType: "node_completed",
      issuedAt: "2026-04-14T00:00:03Z",
      nodeId: "intent_parser",
      payload: { nodeId: "intent_parser" },
      checkpoint: { state: "running", eventSeq: 4 },
    },
    {
      eventId: "evt-005",
      runId: "run_1",
      threadId: "thread_1",
      eventSeq: 5,
      eventType: "run_completed",
      issuedAt: "2026-04-14T00:00:04Z",
      nodeId: null,
      payload: { status: "ok" },
      checkpoint: { state: "completed", eventSeq: 5 },
    },
  ],
  nodes: {},
  finalState: { state: "completed", eventSeq: 5, lastEventId: "evt-005" },
  failureContext: null,
  totalEvents: 5,
};

test("buildViewModel normalizes input and emits deterministic queryKey", () => {
  const vm = buildViewModel(sampleHistory);
  assert.equal(vm.queryKey, "run_1:thread_1:completed");
  assert.equal(vm.totalEvents, 5);
  assert.equal(vm.events.length, 5);
});

test("renderHistoryMode returns html for step-card mode", () => {
  const vm = buildViewModel(sampleHistory);
  const output = renderHistoryMode(vm, VIEW_MODES.STEP_CARD);
  assert.match(output, /step-card-mode/);
  assert.match(output, /intent_parser/);
});

test("renderHistoryMode returns html for dag mode", () => {
  const vm = buildViewModel(sampleHistory);
  const output = renderHistoryMode(vm, VIEW_MODES.DAG);
  assert.match(output, /dag-mode/);
  assert.match(output, /intent_parser/);
});

test("renderHistoryMode returns html for log mode", () => {
  const vm = buildViewModel(sampleHistory);
  const output = renderHistoryMode(vm, VIEW_MODES.LOG);
  assert.match(output, /log-mode/);
  assert.match(output, /node_started/);
});
