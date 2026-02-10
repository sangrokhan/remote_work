# LLM Self-Correction and Structural Chunking

This PR implements an LLM feedback loop for JSON parsing errors and enhances document chunking to respect logical boundaries (sentences and headers).

## Changes

### 1. LLM Self-Correction (Retry Mechanism)
- Added a 1-time automatic retry logic in `GraphPipeline.extract_triples` and `OntologyDiscovery.discover`.
- If the initial JSON parsing fails, the pipeline now sends the malformed text back to the LLM with a correction prompt, allowing the model to "fix itself."
- Added caution notes regarding small model formatting issues.

### 2. Structural Document Chunking
- Updated `GraphPipeline.chunk_document` to use `MarkdownNodeParser` for `.md` files.
- This ensures chunks are split at logical headers and sections rather than arbitrary character limits, preventing mid-sentence cuts.
- Defaults to sentence-level splitting for other file types to maintain data integrity.

### 3. Advanced JSON Repair
- Added regex to fix **unquoted keys** (e.g., `{ key: "val" }`), addressing "expecting property name" errors.
- Added **Javascript comment stripping** (`//` and `/* */`) to handle models that include non-JSON text in their blocks.
- Improved newline-in-string handling to prevent "unterminated string" errors.

## Impact
- Significantly higher success rate when using smaller models (e.g., 7B/8B params) that struggle with strict JSON formatting.
- Better data extraction quality by ensuring text chunks contain complete logical context.
