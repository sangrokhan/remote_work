# Data Drift Analysis Agents: Requirements & Architecture

## 1. Overview
This system is designed to monitor cellular network statistical data for "data drift"—significant deviations in statistical properties over time. The system utilizes a multi-agent architecture to handle the complexity of hundreds of differing data columns (metrics).

## 2. Data Profile & Ingestion Requirements
*   **Source**: Cellular network statistics.
*   **Frequency**: Collected every 5 to 15 minutes.
*   **Dimensions**: Hundreds of columns per record.
*   **Grouping**: Data is logically grouped by a "Group Name" (e.g., Cell ID, Region, Network Slice).
*   **Data Types**:
    1.  **Binning Type**: Histogram-like data distributions (e.g., Signal Strength bins).
    2.  **Averaged Float**: Continuous variables averaged over the collection period (e.g., Average Throughput (Mbps), Latency).
    3.  **Ratio Type**: Percentages or fractions (e.g., Drop Rate, Handover Success Rate).

## 3. Drift Detection Methodology
Different strategies are required for different data types:

| Data Type | Recommended Statistical Test | Drift Metric |
| :--- | :--- | :--- |
| **Binning (Distributions)** | **Population Stability Index (PSI)** or **KL Divergence** | Measures shift in the shape of the distribution. PSI > 0.1 indicates slight drift; > 0.25 indicates significant drift. |
| **Averaged Float** | **Z-Score** / **Standard Deviation Shift** | Measures how far the new mean is from the baseline mean in terms of standard deviations. |
| **Ratio** | **Chi-Square Test** or **Proportion Difference** | Statistically significant difference in success/failure rates. |

## 4. Agentic Architecture
The pipeline is composed of specialized agents:

### A. Data Profiler Agent (The Librarian)
*   **Role**: Inspects incoming metadata.
*   **Task**: Automatically categorizes columns into Binning, Averaged, or Ratio types if not explicitly provided.
*   **Output**: A `Schema Map` json linking columns to their data types.

### B. Statistical Analyst Agent (The Mathematician)
*   **Role**: Computes drift metrics.
*   **Task**:
    1.  Maintains a "Baseline" distribution/stat for each group and column (e.g., sliding window of last 24 hours or 7 days).
    2.  Compares incoming batch (last 5-15 mins) against the Baseline.
    3.  Calculates PSI, Z-Scores, etc.

### C. Insight & Alerting Agent (The Watchdog)
*   **Role**: Interprets results and alerts.
*   **Task**:
    1.  Filters noise (e.g., ignores minor drifts known to be daily seasonality).
    2.  Aggregates drifts (e.g., "All cells in Region X show drift in Throughput").
    3.  Generates a drift report.

## 5. Pipeline Workflow
1.  **Ingest**: Load batch csv/parquet.
2.  **Group**: Split data by "Group Name".
3.  **Profile**: Profiler Agent validates schema.
4.  **Analyze**: Statistical Agent runs tests against Baseline.
5.  **Detect**: Compute Drift Scores.
6.  **Report**: Alerting Agent outputs findings.

## 6. Technical Stack Recommendation
*   **Language**: Python
*   **Libraries**: `pandas`, `numpy`, `scipy` (stats), `alibi-detect` (optional specific drift lib).
*   **Storage**: Time-series database or Parquet files for Baseline storage.
