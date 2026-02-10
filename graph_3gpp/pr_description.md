# Fix for JSON Property Quoting and Unterminated String Errors

This PR provides advanced JSON cleaning to resolve "expecting property name enclosed in double quotes" and "unterminated string" errors occurring during graph construction.

## Changes

### 1. Advanced JSON "Healing" Logic
- Updated `_parse_json` in `GraphPipeline` and `OntologyDiscovery`.
- **Newline Stripping**: Automatically replaces literal newlines within double-quoted strings with spaces. This directly fixes "unterminated string" errors caused by LLMs spreading a single string across multiple lines.
- **Property Quoting Fix**: Specifically targets and converts single-quoted keys (e.g., `'id': 1`) to valid double-quoted JSON keys.
- **Global Quoting Fallback**: Implements a last-resort recovery that attempts to swap all single quotes for double quotes if the parser explicitly flags a property quoting error.

### 2. Maintained Robustness
- Keeps previous improvements: `raw_decode` for "extra data" handling, backslash escaping, and missing comma insertion.

## Impact
- Resolves the most persistent JSON parsing failures reported during large-scale graph generation.
- Increases system autonomy by allowing the pipeline to self-correct common structural LLM errors.
