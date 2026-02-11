# Separate Ontology Files and Schema Enforcement

This PR separates node and relation types into distinct files and adds CLI support for enforcing specific schemas during extraction and discovery.

## Changes

### 1. Separate Ontology Files
- Created `node_types.json` and `relation_types.json` from the original `ontology.json`.
- Updated `GraphPipeline` to support loading these separate files via the constructor and new CLI arguments.

### 2. CLI Argument Enhancements
- Added `--node-types` and `--relation-types` to `graph_pipeline.py`.
- Added `--schema` to `ontology_discovery.py` to allow enforcing a set of valid node labels during the discovery process.

### 3. Documentation
- Updated `README.md` with instructions on how to use the new CLI arguments for custom schema enforcement.

## Impact
- Provides more granular control over the ontology used for triple extraction.
- Improves consistency in ontology discovery by allowing users to pre-define valid entity categories.
