import os
import json
import logging
import re
import argparse
import sys
from typing import List, Dict
from llama_index.llms.openai_like import OpenAILike
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
DEFAULT_ONTOLOGY_PATH = os.getenv("ONTOLOGY_PATH", "ontology.json")

class OntologyDiscovery:
    def __init__(self):
        self.llm = OpenAILike(
            api_base=LLM_API_BASE,
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
            is_chat_model=True,
            timeout=LLM_TIMEOUT
        )

    def _get_samples(self, text: str, file_path: str, sample_size: int = 3000, max_samples: int = 5) -> List[str]:
        """Extracts samples from text. If MD, uses headers; otherwise uses spatial sampling."""
        
        # Markdown-aware sampling
        if file_path.lower().endswith(('.md', '.markdown')):
            logger.info("Markdown detected. Sampling by headers...")
            # Split by headers (e.g., # Section)
            sections = re.split(r'\n(?=#+ )', text)
            if len(sections) > 1:
                # Filter out very short sections and take up to max_samples
                valid_sections = [s for s in sections if len(s.strip()) > 50]
                
                # If we have many sections, pick them across the document
                if len(valid_sections) > max_samples:
                    indices = [int(i * (len(valid_sections) - 1) / (max_samples - 1)) for i in range(max_samples)]
                    return [valid_sections[i][:sample_size] for i in indices]
                return [s[:sample_size] for s in valid_sections]

        # Fallback to spatial sampling for plain text or unstructured MD
        if len(text) <= sample_size * 2:
            return [text]
        
        samples = []
        # Beginning
        samples.append(text[:sample_size])
        
        # Middle points
        if max_samples > 2:
            for i in range(1, max_samples - 1):
                pos = int(len(text) * (i / (max_samples - 1)))
                samples.append(text[pos - (sample_size // 2) : pos + (sample_size // 2)])
        
        # End
        samples.append(text[-sample_size:])
        
        return samples

    def discover(self, file_path: str) -> dict:
        """Analyzes multiple samples of a document to discover its underlying ontology."""
        logger.info(f"Analyzing {file_path} for ontology discovery...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            samples = self._get_samples(text, file_path)
            aggregated_ontology = {"node_types": [], "relationship_types": []}
            
            prompt_template = ONTOLOGY_DISCOVERY_PROMPT.replace("\\n", "\n")

            for i, sample in enumerate(samples):
                logger.info(f"Processing sample {i+1}/{len(samples)}...")
                prompt = prompt_template.format(text=sample)
                
                response = self.llm.complete(prompt)
                output_text = response.text.strip()
                
                json_match = re.search(r"(\{.*\})", output_text, re.DOTALL)
                if json_match:
                    try:
                        sample_ontology = json.loads(json_match.group(1))
                        self._merge_ontologies(aggregated_ontology, sample_ontology)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON from sample {i+1}")
                else:
                    logger.warning(f"No JSON block found in sample {i+1}")

            return aggregated_ontology

        except Exception as e:
            logger.error(f"Ontology discovery failed: {e}")
            return {}

    def _merge_ontologies(self, base: dict, new: dict):
        """Merges new ontology findings into the base ontology, avoiding duplicates."""
        # Merge Nodes
        existing_node_labels = {n['label'].lower(): n for n in base['node_types']}
        for node in new.get('node_types', []):
            label_lower = node['label'].lower()
            if label_lower not in existing_node_labels:
                base['node_types'].append(node)
                existing_node_labels[label_lower] = node
        
        # Merge Relationships
        existing_rel_keys = {(r['source'].lower(), r['type'].lower(), r['target'].lower()) for r in base['relationship_types']}
        for rel in new.get('relationship_types', []):
            rel_key = (rel['source'].lower(), rel['type'].lower(), rel['target'].lower())
            if rel_key not in existing_rel_keys:
                base['relationship_types'].append(rel)
                existing_rel_keys.add(rel_key)

    def save_ontology(self, ontology: dict, output_path: str):
        """Saves the discovered ontology to a file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, indent=2)
        logger.info(f"Ontology saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="3GPP Ontology Discovery Tool (Markdown Aware)")
    parser.add_argument("file", help="Path to the document (txt or md) to analyze")
    parser.add_argument("-o", "--output", help="Path to save the generated ontology", default=DEFAULT_ONTOLOGY_PATH)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
        
    discovery = OntologyDiscovery()
    result = discovery.discover(args.file)
    
    if result and (result['node_types'] or result['relationship_types']):
        discovery.save_ontology(result, args.output)
        print(f"Successfully discovered {len(result['node_types'])} nodes and {len(result['relationship_types'])} relations.")
    else:
        logger.error("Failed to discover a valid ontology.")
        sys.exit(1)

if __name__ == "__main__":
    main()
