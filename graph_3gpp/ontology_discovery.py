import os
import json
import logging
import re
import argparse
import sys
from typing import List, Dict, Set, Any
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Create a dedicated logger for LLM raw outputs
llm_logger = logging.getLogger("llm_output")

class OntologyDiscovery:
    def __init__(self, debug: bool = False, schema_path: str = None):
        self.debug = debug
        self.schema_path = schema_path
        if self.debug:
            llm_logger.setLevel(logging.DEBUG)
        else:
            llm_logger.setLevel(logging.WARNING)

        # Load configurations from environment
        self.api_base = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
        self.api_key = os.getenv("LLM_API_KEY", "EMPTY")
        self.model_name = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "120.0"))
        
        self.chunk_size = int(os.getenv("ONTOLOGY_CHUNK_SIZE", "4000"))
        self.chunk_overlap = int(os.getenv("ONTOLOGY_CHUNK_OVERLAP", "500"))
        self.consolidation_interval = int(os.getenv("ONTOLOGY_CONSOLIDATION_INTERVAL", "10"))
        self.discovery_prompt = os.getenv("ONTOLOGY_DISCOVERY_PROMPT", "").replace("\\n", "\n")
        self.consolidation_prompt = os.getenv("ONTOLOGY_CONSOLIDATION_PROMPT", "").replace("\\n", "\n")
        
        # Initialize components
        self.llm = OpenAILike(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model_name,
            is_chat_model=True,
            timeout=self.timeout
        )
        self.chunker = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.valid_labels = self._load_valid_labels()

    def _load_valid_labels(self) -> Set[str]:
        default_labels = {"NetworkNode", "ProtocolMessage", "Timer", "Procedure", "UserState"}
        if self.schema_path and os.path.exists(self.schema_path):
            try:
                with open(self.schema_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Assume list of objects with "label" or list of strings
                        labels = set()
                        for item in data:
                            if isinstance(item, dict) and "label" in item:
                                labels.add(item["label"])
                            elif isinstance(item, str):
                                labels.add(item)
                        if labels:
                            logger.info(f"Loaded {len(labels)} valid labels from {self.schema_path}")
                            return labels
            except Exception as e:
                logger.error(f"Failed to load schema from {self.schema_path}: {e}")
        return default_labels

    def _consolidate_ontology(self, raw_ontology: dict) -> dict:
        raw_nodes = list(set([n.get('label') for n in raw_ontology.get('node_types', [])]))[:200]
        raw_rels = list(set([f"{r.get('source')} -> {r.get('type')} -> {r.get('target')}" for r in raw_ontology.get('relationship_types', [])]))[:200]

        prompt = self.consolidation_prompt.format(
            nodes=json.dumps(raw_nodes, indent=2), 
            relationships=json.dumps(raw_rels, indent=2)
        )
        
        try:
            response = self.llm.complete(prompt)
            output_text = response.text.strip()
            llm_logger.debug(f"--- Raw LLM Output (Consolidation) ---\n{output_text}\n---------------------------")
            
            consolidated = self._parse_json(output_text)
            
            # Post-processing: Enforce the valid types strictly
            valid_labels = self.valid_labels
            
            # 1. Clean node_types
            if "node_types" in consolidated:
                consolidated["node_types"] = [
                    n for n in consolidated["node_types"] 
                    if n.get("label") in valid_labels
                ]
            
            # 2. Clean relationship_types (ensure source/target are valid)
            if "relationship_types" in consolidated:
                clean_rels = []
                for r in consolidated["relationship_types"]:
                    if r.get("source") in valid_labels and r.get("target") in valid_labels:
                        clean_rels.append(r)
                consolidated["relationship_types"] = clean_rels
                
            return consolidated
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
        return raw_ontology

    def _merge_ontologies(self, base: dict, new: dict):
        valid_labels = self.valid_labels
        
        existing_node_labels = {n['label'].lower(): n for n in base['node_types']}
        for node in new.get('node_types', []):
            label = node.get('label')
            if label in valid_labels and label.lower() not in existing_node_labels:
                base['node_types'].append(node)
                existing_node_labels[label.lower()] = node
        
        # Dictionary to track existing relationships and their indices for easy updating
        existing_rels = {}
        for i, r in enumerate(base['relationship_types']):
            key = (r['source'].lower(), r['type'].lower(), r['target'].lower())
            existing_rels[key] = i

        for rel in new.get('relationship_types', []):
            if rel.get('source') in valid_labels and rel.get('target') in valid_labels:
                rel_key = (rel['source'].lower(), rel['type'].lower(), rel['target'].lower())
                
                if rel_key in existing_rels:
                    # Merge allowed_properties if they exist
                    idx = existing_rels[rel_key]
                    base_rel = base['relationship_types'][idx]
                    
                    new_props = rel.get('allowed_properties', [])
                    if new_props:
                        base_props = base_rel.get('allowed_properties', [])
                        # Merge sets and convert back to list
                        merged_props = list(set(base_props) | set(new_props))
                        base_rel['allowed_properties'] = merged_props
                else:
                    base['relationship_types'].append(rel)
                    existing_rels[rel_key] = len(base['relationship_types']) - 1

    def save_ontology(self, ontology: dict, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, indent=2)
        logger.info(f"Ontology saved to {output_path}")

def main():
    default_ontology_path = os.getenv("ONTOLOGY_PATH", "ontology.json")
    parser = argparse.ArgumentParser(description="3GPP Ontology Discovery Tool")
    parser.add_argument("file", help="Path to the document to analyze")
    parser.add_argument("-o", "--output", help="Path to save the generated ontology", default=default_ontology_path)
    parser.add_argument("--schema", help="Path to a JSON file containing allowed node types/labels")
    parser.add_argument("--full", action="store_true", help="Perform a full scan of the document")
    parser.add_argument("--debug", action="store_true", help="Print raw LLM outputs for debugging")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
        
    # Initialize with debug flag and schema_path
    discovery = OntologyDiscovery(debug=args.debug, schema_path=args.schema)
    result = discovery.discover(args.file, output_path=args.output, full_scan=args.full)
    
    if result and (result.get('node_types') or result.get('relationship_types')):
        discovery.save_ontology(result, args.output)
        print(f"Successfully discovered {len(result.get('node_types', []))} nodes.")
    else:
        logger.error("Failed to discover a valid ontology.")
        sys.exit(1)

if __name__ == "__main__":
    main()