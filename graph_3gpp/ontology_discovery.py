import os
import json
import logging
import re
import argparse
import sys
from typing import List, Dict, Set
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
    def __init__(self, debug: bool = False):
        self.debug = debug
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
        self.discovery_prompt = os.getenv(
            "ONTOLOGY_DISCOVERY_PROMPT",
            "Extract ontology from text:\n{text}\nJSON Output:"
        ).replace("\\n", "\n")
        
        self.consolidation_prompt = """
You are a Senior Ontology Engineer. I will provide you with a raw list of "Node Types" and "Relationship Types" extracted from a large 3GPP document. 
Your job is to MERGE, CLEAN, and STANDARDIZE them into a single, high-quality Ontology.

Instructions:
1. Merge synonyms (e.g., "UE", "User Equipment", "Terminal" -> "UserEquipment").
2. Standardize relationship names (e.g., "sends_message", "transmitting" -> "SENDS").
3. Remove generic or noisy types (e.g., "Device", "Thing").
4. Keep domain-specific types critical for Root Cause Analysis (e.g., "Timer", "RRC_State", "ProtocolError").

Raw Node Types:
{nodes}

Raw Relationship Types:
{relationships}

Return STRICTLY a JSON object with:
{{
  "node_types": [{{"label": "StandardLabel", "description": "Definition"}}],
  "relationship_types": [{{"source": "SourceLabel", "type": "REL_TYPE", "target": "TargetLabel"}}]
}}
"""
        # Initialize components
        self.llm = OpenAILike(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model_name,
            is_chat_model=True,
            timeout=self.timeout
        )
        self.chunker = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

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

    def discover(self, file_path: str, output_path: str = None, full_scan: bool = False) -> dict:
        """Analyzes a document to discover its underlying ontology."""
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

            aggregated_ontology = {"node_types": [], "relationship_types": []}

            for i, sample in enumerate(samples):
                logger.info(f"Processing chunk {i+1}/{len(samples)}...")
                prompt = self.discovery_prompt.format(text=sample)
                
                try:
                    response = self.llm.complete(prompt)
                    output_text = response.text.strip()
                    
                    # Debug log the raw output using the dedicated logger
                    llm_logger.debug(f"--- Raw LLM Output (Chunk {i+1}) ---\n{output_text}\n---------------------------")
                    
                    json_match = re.search(r"(\{.*\})", output_text, re.DOTALL)
                    if json_match:
                        sample_ontology = json.loads(json_match.group(1))
                        self._merge_ontologies(aggregated_ontology, sample_ontology)
                        
                        # Save after each chunk
                        if output_path:
                            self.save_ontology(aggregated_ontology, output_path)
                        
                        # Consolidate every N chunks
                        if (i + 1) % self.consolidation_interval == 0:
                            logger.info(f"Consolidating ontology after {i+1} chunks...")
                            aggregated_ontology = self._consolidate_ontology(aggregated_ontology)
                            if output_path:
                                self.save_ontology(aggregated_ontology, output_path)
                    else:
                        logger.warning(f"No JSON block found in chunk {i+1}. Enable --debug to see raw output.")
                except Exception as e:
                    logger.warning(f"Failed to process chunk {i+1}: {e}")

            if full_scan:
                logger.info("Consolidating final full scan results...")
                return self._consolidate_ontology(aggregated_ontology)
            
            return aggregated_ontology

        except Exception as e:
            logger.error(f"Ontology discovery failed: {e}")
            return {}

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
            
            json_match = re.search(r"(\{.*\})", output_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
        return raw_ontology

    def _merge_ontologies(self, base: dict, new: dict):
        existing_node_labels = {n['label'].lower(): n for n in base['node_types']}
        for node in new.get('node_types', []):
            if node['label'].lower() not in existing_node_labels:
                base['node_types'].append(node)
                existing_node_labels[node['label'].lower()] = node
        
        existing_rel_keys = {(r['source'].lower(), r['type'].lower(), r['target'].lower()) for r in base['relationship_types']}
        for rel in new.get('relationship_types', []):
            rel_key = (rel['source'].lower(), rel['type'].lower(), rel['target'].lower())
            if rel_key not in existing_rel_keys:
                base['relationship_types'].append(rel)
                existing_rel_keys.add(rel_key)

    def save_ontology(self, ontology: dict, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, indent=2)
        logger.info(f"Ontology saved to {output_path}")

def main():
    default_ontology_path = os.getenv("ONTOLOGY_PATH", "ontology.json")
    parser = argparse.ArgumentParser(description="3GPP Ontology Discovery Tool")
    parser.add_argument("file", help="Path to the document to analyze")
    parser.add_argument("-o", "--output", help="Path to save the generated ontology", default=default_ontology_path)
    parser.add_argument("--full", action="store_true", help="Perform a full scan of the document")
    parser.add_argument("--debug", action="store_true", help="Print raw LLM outputs for debugging")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
        
    # Initialize with debug flag
    discovery = OntologyDiscovery(debug=args.debug)
    result = discovery.discover(args.file, output_path=args.output, full_scan=args.full)
    
    if result and (result.get('node_types') or result.get('relationship_types')):
        discovery.save_ontology(result, args.output)
        print(f"Successfully discovered {len(result.get('node_types', []))} nodes.")
    else:
        logger.error("Failed to discover a valid ontology.")
        sys.exit(1)

if __name__ == "__main__":
    main()