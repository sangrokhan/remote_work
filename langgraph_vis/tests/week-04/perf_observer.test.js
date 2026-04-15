import test from "node:test";
import assert from "node:assert/strict";

import { PerfObserver } from "./perf_observer.js";

test("perf observer computes p95 render latency and memory peak", () => {
  const observer = new PerfObserver({
    name: "poC",
    thresholds: {
      renderP95Ms: 300,
      throughputEventsPerSec: 200,
      memoryPeakMb: 512,
      hydrateSuccessRate: 99.5,
      failureCoveragePercent: 100,
    },
  });

  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100].forEach((ms) => observer.recordRenderMs(ms));
  observer.recordThroughputEvents({ eventCount: 200, elapsedMs: 1000 });
  observer.recordThroughputEvents({ eventCount: 250, elapsedMs: 1000 });
  observer.recordMemorySample(128.5);
  observer.recordMemorySample(129.0);
  observer.recordMemorySample(64.2);
  observer.recordHydrate({ success: true });
  observer.recordHydrate({ success: true });
  observer.recordHydrate({ success: false });

  const snapshot = observer.getSnapshot();
  assert.equal(snapshot.renderP95Ms, 100);
  assert.equal(snapshot.memoryPeakMb, 129);
  assert.equal(snapshot.hydrateSuccessRate, 66.66666666666666);
  assert.equal(observer.isPass(), false);
});

test("perf observer passes with good thresholds", () => {
  const observer = new PerfObserver({
    thresholds: {
      renderP95Ms: 300,
      throughputEventsPerSec: 150,
      memoryPeakMb: 10000,
      hydrateSuccessRate: 90,
      failureCoveragePercent: 100,
    },
  });

  [50, 60, 70, 80, 90, 100].forEach((ms) => observer.recordRenderMs(ms));
  observer.recordThroughputEvents({ eventCount: 300, elapsedMs: 1000 });
  observer.recordMemorySample(5);
  observer.recordHydrate({ success: true });
  observer.recordHydrate({ success: true });

  const snapshot = observer.getSnapshot();
  assert.equal(observer.isPass(), true);
  assert.equal(snapshot.throughputEventsPerSec > 150, true);
});
