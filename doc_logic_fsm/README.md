# Document-based Logic Extraction & FSM Construction (doc_logic_fsm)

This module is designed to automatically extract protocol logic from natural language specifications (e.g., 3GPP TS) and synthesize them into Formal Behavioral Models (FSM/EFSM).

## Installation & Setup
1. **Python Version**: Recommended 3.10+ (Tested on 3.12).
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Running the Pipeline
You can run a full check of the extraction and visualization pipeline:
```bash
python tests/test_pipeline.py
```

## Directory Structure
- `docs/`: Raw specification documents (PDF, Text, HTML).
- `preprocessing/`: Scripts for parsing and cleaning raw text into structured formats.
- `nlp_engine/`: Logic extraction using Semantic Role Labeling (SRL) and dependency parsing.
- `fsm_core/`: Engine for synthesizing, linking, and managing hierarchical/parallel FSMs.
- `validation/`: Tools for identifying deviations between extracted FSMs and implementation traces.
- `configs/`: Configuration files for different protocol stacks and extraction rules.

## Core Objective
1. **Automated Extraction**: Transform "The UE shall send X and enter state Y" into machine-readable state transitions.
2. **Cross-Reference Resolution**: Link logic across different document versions and sections.
3. **Behavioral Analysis**: Use formal models to detect implementation gaps or security flaws.
