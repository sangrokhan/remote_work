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
load_dotenv(override=True)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Create a dedicated logger for LLM raw outputs
llm_logger = logging.getLogger("llm_output")

class OntologyDiscovery:
    def __init__(self, debug: bool = False, node_types_path: str = None, rel_types_path: str = None):
        self.debug = debug
        self.node_types_path = node_types_path
        self.rel_types_path = rel_types_path
        if self.debug:
            llm_logger.setLevel(logging.DEBUG)
        else:
            llm_logger.setLevel(logging.WARNING)

        # Load configurations from environment
        self.api_base = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
        self.api_key = os.getenv("LLM_API_KEY", "EMPTY")
        self.model_name = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "300.0"))
        
        self.chunk_size = int(os.getenv("ONTOLOGY_CHUNK_SIZE", "4000"))
        self.chunk_overlap = int(os.getenv("ONTOLOGY_CHUNK_OVERLAP", "500"))
        self.consolidation_interval = int(os.getenv("ONTOLOGY_CONSOLIDATION_INTERVAL", "10"))
        self.discovery_prompt = os.getenv("ONTOLOGY_DISCOVERY_PROMPT", "")
        self.consolidation_prompt = os.getenv("ONTOLOGY_CONSOLIDATION_PROMPT", "")
        
        # Initialize components
        self.llm = OpenAILike(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model_name,
            is_chat_model=True,
            timeout=self.timeout,
            max_tokens=2048
        )
        self.chunker = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        
        # Load valid base types from node_types_path if provided, else use defaults
        self.valid_base_types = self._load_node_labels()
        # Load valid relation types from rel_types_path if provided
        self.valid_rel_types = self._load_rel_types()

    def _load_node_labels(self) -> Set[str]:
        default_labels = {"NetworkNode", "ProtocolMessage", "Timer", "Procedure", "UserState"}
        if self.node_types_path and os.path.exists(self.node_types_path):
            try:
                with open(self.node_types_path, 'r') as f:
                    data = json.load(f)
                    labels = set()
                    for item in data:
                        if isinstance(item, dict) and "label" in item:
                            labels.add(item["label"])
                        elif isinstance(item, str):
                            labels.add(item)
                    if labels:
                        logger.info(f"Loaded {len(labels)} valid base types from {self.node_types_path}")
                        return labels
            except Exception as e:
                logger.error(f"Failed to load node types from {self.node_types_path}: {e}")
        return default_labels

    def _load_rel_types(self) -> Set[str]:
        if self.rel_types_path and os.path.exists(self.rel_types_path):
            try:
                with open(self.rel_types_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        logger.info(f"Loaded {len(data)} valid relation types from {self.rel_types_path}")
                        return set(data)
            except Exception as e:
                logger.error(f"Failed to load relation types from {self.rel_types_path}: {e}")
        return set()

    def _get_samples(self, text: str, file_path: str, sample_size: int = 3000, max_samples: int = 5) -> List[str]:
        """Extracts samples from text. If MD, uses headers; otherwise uses spatial sampling."""
        if file_path.lower().endswith(('.md', '.markdown')):
            logger.info("Markdown detected. Sampling by headers...")
            sections = re.split(r'\n(?=#+ )', text)
            if len(sections) > 1:
                valid_sections = [s for s in sections if len(s.strip()) > 50]
                if len(valid_sections) > max_samples:
                    indices = [int(i * (len(valid_sections) - 1) / (max_samples - 1)) for i in range(max_samples)]
                    return [valid_sections[i][:sample_size] for i in indices]
                return [s[:sample_size] for s in valid_sections]

        if len(text) <= sample_size * 2:
            return [text]
        
        samples = []
        samples.append(text[:sample_size])
        if max_samples > 2:
            for i in range(1, max_samples - 1):
                pos = int(len(text) * (i / (max_samples - 1)))
                samples.append(text[pos - (sample_size // 2) : pos + (sample_size // 2)])
        samples.append(text[-sample_size:])
        return samples

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extracts and parses JSON from LLM response using dirtyjson for robustness."""
        import dirtyjson
        import re
        
        # 1. Find the start and end of the JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            return {}
        
        json_str = text[start:end+1]
        
        try:
            return dirtyjson.loads(json_str)
        except Exception as e:
            logger.error(f"JSON Parsing Error with dirtyjson: {e}")
            try:
                # Basic cleanup
                json_str = re.sub(r'//.*?\n|/\*.*?\*/', '', json_str, flags=re.DOTALL)
                return dirtyjson.loads(json_str)
            except:
                return {}

    def discover(self, file_path: str, output_path: str = None, full_scan: bool = False, retry_count: int = 1) -> dict:
        """Analyzes a document to discover its underlying ontology with self-correction."""
        logger.info(f"Analyzing {file_path} for ontology discovery...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if full_scan:
                logger.info("Performing FULL SCAN of the document. This may take time...")
                doc = Document(text=text)
                chunks = self.chunker.get_nodes_from_documents([doc])
                samples = [chunk.get_content() for chunk in chunks]
            else:
                logger.info("Performing SAMPLE-BASED discovery.")
                samples = self._get_samples(text, file_path)

            aggregated_ontology = {"nodes": [], "relations": []}

            for i, sample in enumerate(samples):
                logger.info(f"Processing chunk {i+1}/{len(samples)}...")
                prompt = self.discovery_prompt.format(text=sample)
                
                try:
                    response = self.llm.complete(prompt)
                    output_text = response.text.strip()
                    llm_logger.debug(f"--- Raw LLM Output (Chunk {i+1}) ---\n{output_text}\n---------------------------")
                    
                    sample_ontology = self._parse_json(output_text)
                    
                    if not sample_ontology and retry_count > 0:
                        logger.warning(f"Initial discovery JSON parsing failed for chunk {i+1}. Attempting self-correction...")
                        correction_prompt = f"The following JSON was invalid. Please fix the formatting and return ONLY the valid JSON object:\n\n{output_text}"
                        retry_response = self.llm.complete(correction_prompt)
                        sample_ontology = self._parse_json(retry_response.text.strip())

                    if sample_ontology:
                        self._merge_ontologies(aggregated_ontology, sample_ontology)
                        if output_path:
                            self.save_ontology(aggregated_ontology, output_path)
                        
                        if (i + 1) % self.consolidation_interval == 0:
                            logger.info(f"Consolidating ontology...")
                            aggregated_ontology = self._consolidate_ontology(aggregated_ontology)
                            if output_path:
                                self.save_ontology(aggregated_ontology, output_path)
                    else:
                        logger.warning(f"No valid JSON block found in chunk {i+1}. Enable --debug to see raw output.")
                except Exception as e:
                    logger.warning(f"Failed to process chunk {i+1}: {e}")

            logger.info("Finalizing ontology with strict categorization...")
            return self._consolidate_ontology(aggregated_ontology)

        except Exception as e:
            logger.error(f"Ontology discovery failed: {e}")
            return {}

    def _consolidate_ontology(self, raw_ontology: dict) -> dict:
        # Prepare the data for the LLM to consolidate
        raw_nodes = []
        for n in raw_ontology.get('nodes', []):
             raw_nodes.append({
                 "label": n.get("label"),
                 "node_type": n.get("node_type"),
                 "properties": n.get("properties")
             })
        
        raw_rels = raw_ontology.get('relations', [])

        prompt = self.consolidation_prompt.format(
            nodes=json.dumps(raw_nodes, indent=2)
        )
        
        try:
            response = self.llm.complete(prompt)
            output_text = response.text.strip()
            llm_logger.debug(f"--- Raw LLM Output (Consolidation) ---\n{output_text}\n---------------------------")
            
            consolidated = self._parse_json(output_text)
            
            # Post-processing: Validate node_types strictly
            valid_base_types = self.valid_base_types
            
            # 1. Clean nodes
            if "nodes" in consolidated:
                clean_nodes = []
                for n in consolidated["nodes"]:
                    b_type = n.get("node_type") or n.get("label")
                    if b_type in valid_base_types:
                        clean_nodes.append(n)
                consolidated["nodes"] = clean_nodes
            
            return consolidated
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
        return raw_ontology

    def _merge_ontologies(self, base: dict, new: dict):
        # 1. Merge Nodes (support both old and new keys for transition)
        new_nodes = new.get('nodes', []) or new.get('node_types', [])
        existing_nodes = {n['label'].lower(): i for i, n in enumerate(base.get('nodes', []))}
        
        for node in new_nodes:
            label = node.get('label')
            if label:
                low_label = label.lower()
                if low_label in existing_nodes:
                    idx = existing_nodes[low_label]
                    base_props = base['nodes'][idx].get('properties', {})
                    new_props = node.get('properties', {})
                    if isinstance(base_props, dict) and isinstance(new_props, dict):
                        base_props.update(new_props)
                        base['nodes'][idx]['properties'] = base_props
                else:
                    if 'nodes' not in base: base['nodes'] = []
                    base['nodes'].append(node)
                    existing_nodes[low_label] = len(base['nodes']) - 1
        
        # 2. Merge Relations
        new_rels = new.get('relations', []) or new.get('relationship_types', [])
        if "relations" not in base:
            base["relations"] = []
            
        existing_rels = {(r.get("source"), r.get("type"), r.get("target")): i 
                         for i, r in enumerate(base["relations"]) if isinstance(r, dict)}
        
        for rel in new_rels:
            if isinstance(rel, dict):
                key = (rel.get("source"), rel.get("type"), rel.get("target"))
                if key in existing_rels:
                    idx = existing_rels[key]
                    base_props = base['relations'][idx].get('properties', {})
                    new_props = rel.get('properties', {})
                    if isinstance(base_props, dict) and isinstance(new_props, dict):
                        base_props.update(new_props)
                        base['relations'][idx]['properties'] = base_props
                else:
                    base['relations'].append(rel)
                    existing_rels[key] = len(base['relations']) - 1
            else:
                # If it's just a string, add it if not present
                if rel not in base["relations"]:
                    base["relations"].append(rel)

    def save_ontology(self, ontology: dict, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, indent=2)
        logger.info(f"Ontology saved to {output_path}")

def main():
    default_ontology_path = os.getenv("ONTOLOGY_PATH", "ontology.json")
    parser = argparse.ArgumentParser(description="3GPP Ontology Discovery Tool")
    parser.add_argument("file", help="Path to the document to analyze")
    parser.add_argument("-o", "--output", help="Path to save the generated ontology", default=default_ontology_path)
    parser.add_argument("--node-types", help="Path to a JSON file containing allowed node types/labels", default="node_types.json")
    parser.add_argument("--relation-types", help="Path to a JSON file containing allowed relationship types", default="relation_types.json")
    parser.add_argument("--full", action="store_true", help="Perform a full scan of the document")
    parser.add_argument("--debug", action="store_true", help="Print raw LLM outputs for debugging")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
        
    discovery = OntologyDiscovery(debug=args.debug, node_types_path=args.node_types, rel_types_path=args.relation_types)
    result = discovery.discover(args.file, output_path=args.output, full_scan=args.full)
    
    if result and (result.get('nodes') or result.get('relations')):
        discovery.save_ontology(result, args.output)
        print(f"Successfully discovered {len(result.get('nodes', []))} nodes.")
    else:
        logger.error("Failed to discover a valid ontology.")
        sys.exit(1)

if __name__ == "__main__":
    main()
