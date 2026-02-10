# Contextual Ontology Filtering for Large Documents

This PR implements a dynamic ontology filtering mechanism to handle technical specifications where the discovered ontology exceeds LLM token limits (e.g., 15k+ lines).

## Changes

### 1. Dynamic Contextual Ontology Filtering
- Added `_get_relevant_ontology_str(text)` method to `GraphPipeline`.
- Scans each text chunk for mentioned Node Labels before calling the LLM.
- Only includes relevant Node Types and their associated Relationship Types in the extraction prompt.

### 2. Token Optimization
- **Description Stripping**: Automatically removes long "description" fields from the ontology JSON sent to the LLM.
- **Minification**: Uses compact JSON serialization (no extra whitespace) to further reduce token usage.
- **Safety Bounds**: Implements hard limits on the number of nodes (100) and relationships (150) per prompt.

### 3. Resilience
- Implements a fallback mechanism that provides a subset of the ontology if no direct matches are found in a chunk.
- Improved JSON loading with better error reporting.

## Impact
- Allows processing of documents with massive ontologies that previously caused "Token limit exceeded" errors.
- Reduces cost and latency per LLM call by significantly shrinking prompt size.
