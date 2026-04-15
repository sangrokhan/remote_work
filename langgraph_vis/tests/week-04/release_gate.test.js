import test from "node:test";
import assert from "node:assert/strict";

import { evaluateReleaseGate } from "./release_gate.js";

test("release gate passes when all phase checks satisfy thresholds", () => {
  const result = evaluateReleaseGate({
    metrics: {
      renderP95Ms: 90,
      throughputEventsPerSec: 250,
      memoryPeakMb: 64,
      hydrateAttempts: 200,
      hydrateSuccessRate: 100,
      failureCoveragePercent: 100,
      kpis: {
        hasCanonicalParity: true,
        hasHistoryResponse: true,
        hasFailureContextCoverage: true,
      },
    },
  });

  assert.equal(result.phase0, "PASS");
  assert.equal(result.phase1, "PASS");
  assert.equal(result.phase2, "PASS");
  assert.equal(result.overall, "PASS");
  assert.equal(result.decision, "PROCEED");
});

test("release gate rejects when phase1 KPI fails", () => {
  const result = evaluateReleaseGate({
    metrics: {
      renderP95Ms: 450,
      throughputEventsPerSec: 250,
      memoryPeakMb: 64,
      hydrateAttempts: 1_000,
      hydrateSuccessRate: 97,
      failureCoveragePercent: 100,
      kpis: {
        hasCanonicalParity: true,
        hasHistoryResponse: true,
        hasFailureContextCoverage: true,
      },
    },
    options: {
      thresholds: {
        renderP95Ms: 300,
      },
    },
  });

  assert.equal(result.phase0, "PASS");
  assert.equal(result.phase1, "REJECT");
  assert.equal(result.overall, "REJECT");
  assert.equal(result.decision, "BLOCK");
});
