# Fix for Unquoted Keys and Comments in JSON

This PR enhances the JSON parsing robustness to handle "expecting property name enclosed in double quotes" errors caused by unquoted keys or comments in the LLM output.

## Changes

### 1. Advanced JSON Cleaning Enhancements
- Updated `_parse_json` in `GraphPipeline` and `OntologyDiscovery`.
- **Unquoted Key Repair**: Added regex to detect and automatically quote unquoted property names (e.g., `{ head: "UE" }` -> `{ "head": "UE" }`).
- **Comment Stripping**: Automatically removes Javascript-style line (`//`) and block (`/* */`) comments that LLMs sometimes include in JSON responses.
- **Refined Quoting Logic**: Improved detection and conversion of single-quoted keys.
- **Literal Newline Handling**: Continued support for stripping illegal newlines within strings.

### 2. Resilience
- Maintains previous improvements including `raw_decode` for "extra data" isolation and backslash escaping.

## Impact
- Resolves persistent parsing failures where the LLM produces technically invalid JSON that is still structurally readable.
- Improves the success rate of the graph pipeline when using models prone to non-standard JSON formatting.