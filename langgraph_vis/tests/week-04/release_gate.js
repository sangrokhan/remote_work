import { PERF_DEFAULT_THRESHOLDS } from "./perf_observer.js";

function _safeBool(value) {
  return Boolean(value);
}

function _gateState(pass) {
  return pass ? "PASS" : "REJECT";
}

export function evaluateReleaseGate({metrics, options = {}}) {
  const thresholds = {
    ...PERF_DEFAULT_THRESHOLDS,
    ...options.thresholds,
  };

  const hasCanonicalParity = _safeBool(metrics?.kpis?.hasCanonicalParity ?? true);
  const hasHistoryResponse = _safeBool(metrics?.kpis?.hasHistoryResponse ?? true);
  const hasFailureContextCoverage = _safeBool(metrics?.kpis?.hasFailureContextCoverage ?? false);

  const phase0Checks = {
    canonicalParity: hasCanonicalParity,
    historyResponse: hasHistoryResponse,
  };
  const phase0Pass = Object.values(phase0Checks).every(Boolean);

  const phase1Checks = {
    renderP95: metrics.renderP95Ms !== null && metrics.renderP95Ms <= thresholds.renderP95Ms,
    throughput: metrics.throughputEventsPerSec >= thresholds.throughputEventsPerSec,
    memoryPeak: metrics.memoryPeakMb === null || metrics.memoryPeakMb <= thresholds.memoryPeakMb,
    hydrateSuccessRate: metrics.hydrateAttempts === 0 || metrics.hydrateSuccessRate >= thresholds.hydrateSuccessRate,
  };
  const phase1Pass = Object.values(phase1Checks).every(Boolean);

  const phase2Checks = {
    failureDiagnosticCoverage: hasFailureContextCoverage && metrics.failureCoveragePercent >= thresholds.failureCoveragePercent,
  };
  const phase2Pass = Object.values(phase2Checks).every(Boolean);

  return {
    thresholds,
    checks: { phase0: phase0Checks, phase1: phase1Checks, phase2: phase2Checks },
    phase0: _gateState(phase0Pass),
    phase1: _gateState(phase1Pass),
    phase2: _gateState(phase2Pass),
    overall: _gateState(phase0Pass && phase1Pass && phase2Pass),
    decision: phase0Pass && phase1Pass && phase2Pass ? "PROCEED" : "BLOCK",
  };
}
