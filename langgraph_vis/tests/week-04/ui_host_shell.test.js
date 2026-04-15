import test from "node:test";
import assert from "node:assert/strict";

import { MemoryStorage, UiHostShell } from "./ui_host_shell.js";

test("host shell keeps open/close and mode state via key-persisted storage", () => {
  const storage = new MemoryStorage();
  const shell = new UiHostShell({ runId: "run_1", threadId: "thread_1", storage });

  shell.setMode("log");
  assert.equal(shell.getState().mode, "log");
  assert.equal(shell.getState().isOpen, true);

  shell.close();
  assert.equal(shell.getState().isOpen, false);
  shell.open();
  assert.equal(shell.getState().isOpen, true);

  const remount = new UiHostShell({ runId: "run_1", threadId: "thread_1", storage });
  assert.equal(remount.getState().mode, "log");
  assert.equal(remount.getState().isOpen, true);
});

test("queryKey transition preserves prior mode and defaults for unseen key", () => {
  const storage = new MemoryStorage();
  const shellA = new UiHostShell({ runId: "run_a", threadId: "thread_a", storage });
  shellA.setMode("dag");
  shellA.setQueryKey("run_b:thread_b");
  shellA.setMode("step-card");

  const shellB = new UiHostShell({ runId: "run_b", threadId: "thread_b", storage });
  assert.equal(shellB.getState().mode, "step-card");

  const shellA2 = new UiHostShell({ runId: "run_a", threadId: "thread_a", storage });
  assert.equal(shellA2.getState().mode, "dag");
});

test("mode toggles remain stable over repeated transitions", () => {
  const storage = new MemoryStorage();
  const shell = new UiHostShell({ runId: "run_x", threadId: "thread_x", storage });
  let mode = shell.getState().mode;
  for (let index = 0; index < 10; index += 1) {
    shell.setMode(index % 2 === 0 ? "log" : "step-card");
    mode = shell.getState().mode;
  }
  assert.equal(mode, "step-card");
  assert.equal(shell.getState().isOpen, true);
});
