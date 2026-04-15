const DEFAULT_THRESHOLDS = {
  renderP95Ms: 300,
  throughputEventsPerSec: 200,
  memoryPeakMb: 256,
  hydrateSuccessRate: 99.5,
  failureCoveragePercent: 100,
};

function _toNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function _quantile(values, p) {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }
  const sorted = [...values]
    .map((value) => _toNumber(value, null))
    .filter((value) => value !== null && Number.isFinite(value))
    .sort((a, b) => a - b);

  if (sorted.length === 0) {
    return null;
  }
  const index = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[index];
}

function _safePercent(numerator, denominator) {
  if (!Number.isFinite(_toNumber(numerator)) || !Number.isFinite(_toNumber(denominator)) || denominator <= 0) {
    return 0;
  }
  return (numerator / denominator) * 100;
}

export class PerfObserver {
  constructor({name = "default", thresholds = {}} = {}) {
    this.name = String(name || "default");
    this.thresholds = { ...DEFAULT_THRESHOLDS, ...thresholds };
    this.renderSamplesMs = [];
    this.throughputSamples = [];
    this.memorySamplesMb = [];
    this.hydrateAttempts = 0;
    this.hydrateSuccess = 0;
    this.eventSamples = 0;
  }

  recordRenderMs(ms) {
    const value = _toNumber(ms, null);
    if (value === null || value < 0) {
      return;
    }
    this.renderSamplesMs.push(value);
  }

  recordThroughputEvents({eventCount, elapsedMs} = {}) {
    const count = _toNumber(eventCount, null);
    const elapsed = _toNumber(elapsedMs, null);
    if (count === null || elapsed === null || elapsed <= 0 || count < 0) {
      return;
    }
    const eps = (count / (elapsed / 1000));
    this.throughputSamples.push(eps);
    this.eventSamples += count;
  }

  recordMemorySample(value = null) {
    const sample = value == null ? _toNumber(process.memoryUsage().heapUsed, null) / (1024 * 1024) : _toNumber(value, null);
    if (sample === null || sample < 0) {
      return;
    }
    this.memorySamplesMb.push(sample);
  }

  recordHydrate({success = false} = {}) {
    this.hydrateAttempts += 1;
    if (success) {
      this.hydrateSuccess += 1;
    }
  }

  getSnapshot() {
    const p95RenderMs = _quantile(this.renderSamplesMs, 95);
    const maxMemoryMb = this.memorySamplesMb.length ? Math.max(...this.memorySamplesMb) : null;
    const avgThroughput = this.throughputSamples.length
      ? this.throughputSamples.reduce((sum, value) => sum + value, 0) / this.throughputSamples.length
      : 0;
    return {
      name: this.name,
      renderP95Ms: p95RenderMs,
      memoryPeakMb: maxMemoryMb,
      throughputEventsPerSec: avgThroughput,
      hydrateAttempts: this.hydrateAttempts,
      hydrateSuccessRate: _safePercent(this.hydrateSuccess, this.hydrateAttempts),
      renderedEvents: this.eventSamples,
      sampleCount: this.renderSamplesMs.length,
    };
  }

  isPass() {
    const snapshot = this.getSnapshot();
    return (
      snapshot.renderP95Ms !== null &&
      snapshot.renderP95Ms <= this.thresholds.renderP95Ms &&
      (snapshot.hydrateAttempts === 0 || snapshot.hydrateSuccessRate >= this.thresholds.hydrateSuccessRate) &&
      (snapshot.memoryPeakMb === null || snapshot.memoryPeakMb <= this.thresholds.memoryPeakMb)
    );
  }
}

export { DEFAULT_THRESHOLDS as PERF_DEFAULT_THRESHOLDS };
