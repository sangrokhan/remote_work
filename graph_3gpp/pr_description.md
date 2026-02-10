# Enhanced Robust JSON Parsing for Graph Construction

This PR significantly improves the JSON parsing resilience in the graph pipeline to handle common LLM formatting errors encountered during technical document processing.

## Changes

### 1. Advanced JSON Cleaning Logic
- Enhanced `_parse_json` method in `GraphPipeline` and `OntologyDiscovery`.
- **Trailing Comma Removal**: Automatically strips illegal trailing commas in objects and arrays (e.g., `{"a": 1, }`).
- **Improved Delimiter Recovery**: Better detection and insertion of missing commas between fields.
- **Single Quote Fallback**: Added a secondary parsing attempt that tries to fix non-standard single-quoted JSON markers.
- **Refined Block Extraction**: Improved logic for isolating the outermost JSON object from conversational LLM noise.

### 2. Improved Diagnostics
- Added more granular logging for JSON parsing errors.
- Failed JSON strings are now logged at the `DEBUG` level to facilitate rapid troubleshooting of new LLM failure modes.

## Impact
- Resolves "Expecting ',' delimiter" and "Invalid \escape" errors occurring during the graph construction phase.
- Increases the overall success rate of triple extraction when using smaller or more creative LLM models.