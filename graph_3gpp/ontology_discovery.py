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

# LLM Configuration
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120.0"))

# Discovery Configuration
ONTOLOGY_DISCOVERY_PROMPT = os.getenv(
    "ONTOLOGY_DISCOVERY_PROMPT",
    "Extract ontology from text:\n{text}\nJSON Output:"
)
CONSOLIDATION_PROMPT = """
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
DEFAULT_ONTOLOGY_PATH = os.getenv("ONTOLOGY_PATH", "ontology.json")
ONTOLOGY_CHUNK_SIZE = int(os.getenv("ONTOLOGY_CHUNK_SIZE", "4000"))
ONTOLOGY_CHUNK_OVERLAP = int(os.getenv("ONTOLOGY_CHUNK_OVERLAP", "500"))

class OntologyDiscovery:
    def __init__(self):
        self.llm = OpenAILike(
            api_base=LLM_API_BASE,
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
            is_chat_model=True,
            timeout=LLM_TIMEOUT
        )
        self.chunker = SentenceSplitter(chunk_size=ONTOLOGY_CHUNK_SIZE, chunk_overlap=ONTOLOGY_CHUNK_OVERLAP)

    def _get_samples(self, text: str, file_path: str, sample_size: int = 3000, max_samples: int = 5) -> List[str]:
        """Extracts samples from text. If MD, uses headers; otherwise uses spatial sampling."""
        
        # Markdown-aware sampling
        if file_path.lower().endswith(('.md', '.markdown')):
            logger.info("Markdown detected. Sampling by headers...")
            sections = re.split(r'\n(?=#+ )', text)
            if len(sections) > 1:
                valid_sections = [s for s in sections if len(s.strip()) > 50]
                if len(valid_sections) > max_samples:
                    indices = [int(i * (len(valid_sections) - 1) / (max_samples - 1)) for i in range(max_samples)]
                    return [valid_sections[i][:sample_size] for i in indices]
                return [s[:sample_size] for s in valid_sections]

        # Fallback to spatial sampling
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

    def discover(self, file_path: str, full_scan: bool = False) -> dict:
        """Analyzes a document to discover its underlying ontology.
        
        Args:
            file_path: Path to the document.
            full_scan: If True, scans the entire document (Map-Reduce). If False, uses sampling.
        """
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
            prompt_template = ONTOLOGY_DISCOVERY_PROMPT.replace("\\n", "\n")

            # MAP PHASE
            for i, sample in enumerate(samples):
                logger.info(f"Processing chunk {i+1}/{len(samples)}...")
                prompt = prompt_template.format(text=sample)
                
                try:
                    response = self.llm.complete(prompt)
                    output_text = response.text.strip()
                    
                    json_match = re.search(r"(\{.*\})", output_text, re.DOTALL)
                    if json_match:
                        sample_ontology = json.loads(json_match.group(1))
                        self._merge_ontologies(aggregated_ontology, sample_ontology)
                    else:
                        logger.warning(f"No JSON block found in chunk {i+1}")
                except Exception as e:
                    logger.warning(f"Failed to process chunk {i+1}: {e}")

            # REDUCE PHASE (Consolidation)
            if full_scan:
                logger.info("Consolidating full scan results...")
                return self._consolidate_ontology(aggregated_ontology)
            
            return aggregated_ontology

        except Exception as e:
            logger.error(f"Ontology discovery failed: {e}")
            return {}

    def _consolidate_ontology(self, raw_ontology: dict) -> dict:
        """Uses LLM to merge and clean the aggregated ontology."""
        
        # Flatten lists for the prompt
        raw_nodes = [n.get('label') for n in raw_ontology.get('node_types', [])]
        raw_rels = [f"{r.get('source')} -> {r.get('type')} -> {r.get('target')}" for r in raw_ontology.get('relationship_types', [])]
        
        # Limit the lists to avoid context overflow (if massive)
        raw_nodes = list(set(raw_nodes))[:200] 
        raw_rels = list(set(raw_rels))[:200]

        prompt = CONSOLIDATION_PROMPT.format(
            nodes=json.dumps(raw_nodes, indent=2), 
            relationships=json.dumps(raw_rels, indent=2)
        )
        
        try:
            response = self.llm.complete(prompt)
            output_text = response.text.strip()
            
            json_match = re.search(r"(\{.*\})", output_text, re.DOTALL)
            if json_match:
                final_ontology = json.loads(json_match.group(1))
                logger.info("Ontology successfully consolidated.")
                return final_ontology
            else:
                logger.error("Failed to parse consolidated ontology JSON.")
                return raw_ontology # Return raw as fallback

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            return raw_ontology

    def _merge_ontologies(self, base: dict, new: dict):
        """Merges new ontology findings into the base ontology."""
        existing_node_labels = {n['label'].lower(): n for n in base['node_types']}
        for node in new.get('node_types', []):
            label_lower = node['label'].lower()
            if label_lower not in existing_node_labels:
                base['node_types'].append(node)
                existing_node_labels[label_lower] = node
        
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
    parser = argparse.ArgumentParser(description="3GPP Ontology Discovery Tool")
    parser.add_argument("file", help="Path to the document to analyze")
    parser.add_argument("-o", "--output", help="Path to save the generated ontology", default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--full", action="store_true", help="Perform a full scan of the document (slower but more accurate)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
        
    discovery = OntologyDiscovery()
    result = discovery.discover(args.file, full_scan=args.full)
    
    if result and (result.get('node_types') or result.get('relationship_types')):
        discovery.save_ontology(result, args.output)
        print(f"Successfully discovered {len(result.get('node_types', []))} nodes and {len(result.get('relationship_types', []))} relations.")
    else:
        logger.error("Failed to discover a valid ontology.")
        sys.exit(1)

if __name__ == "__main__":
    main()