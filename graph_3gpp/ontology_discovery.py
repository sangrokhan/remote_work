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
        """Extracts and parses JSON from LLM response, with robust cleaning."""
        import re
        # Find the first {
        start = text.find('{')
        if start == -1:
            return {}
        
        json_str = text[start:]
        
        # 1. Handle common LLM JSON formatting issues
        # Remove literal newlines within double-quoted strings
        def replace_newlines(match):
            return match.group(0).replace('\n', ' ').replace('\r', ' ')
        
        json_str = re.sub(r'"[^"]*"', replace_newlines, json_str, flags=re.DOTALL)

        # 2. Strip comments (Javascript style // or /* */)
        json_str = re.sub(r'//.*?\n|/\*.*?\*/', '', json_str, flags=re.DOTALL)

        # 3. Basic cleaning
        # Escape backslashes that are NOT part of a valid JSON escape sequence
        json_str = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', json_str)
        
        # 4. Fix Quoting and Delimiters
        # Fix "expecting property name enclosed in double quotes" (unquoted keys)
        # { head: "UE" } -> { "head": "UE" }
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Fix single quotes for keys: 'property': value -> "property": value
        json_str = re.sub(r"([{,]\s*)'([^'\" ]+)'\s*:", r'\1"\2":', json_str)

        # Try to fix missing commas between fields
        json_str = re.sub(r'("(?:\\["\\/bfnrtu]|[^"\\])*")\s+(")', r'\1, \2', json_str)
        json_str = re.sub(r'([}\]])\s+(")', r'\1, \2', json_str)
        json_str = re.sub(r'(\b\d+\b|true|false|null)\s+(")', r'\1, \2', json_str)
        
        # 5. Remove trailing commas (illegal in standard JSON)
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

        try:
            # Use raw_decode to handle "extra data" after the JSON object
            decoder = json.JSONDecoder()
            obj, index = decoder.raw_decode(json_str)
            return obj
        except json.JSONDecodeError as e:
            # Final desperate attempt: replace remaining single quotes with double quotes
            # ONLY if the error seems related to quoting
            if "expecting property name" in str(e) or "enclosed in double quotes" in str(e):
                try:
                    # Naive replacement of single quotes with double quotes for the whole string
                    fallback_json = json_str.replace("'", '"')
                    obj, index = decoder.raw_decode(fallback_json)
                    return obj
                except:
                    pass
            
            logger.error(f"JSON Parsing Error: {e}")
            llm_logger.debug(f"Failed JSON string: {json_str[:500]}...")
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

            aggregated_ontology = {"node_types": [], "relationship_types": []}

            for i, sample in enumerate(samples):
                logger.info(f"Processing chunk {i+1}/{len(samples)}...")
                prompt = self.discovery_prompt.format(text=sample)
                
                try:
                    response = self.llm.complete(prompt)
                    output_text = response.text.strip()
                    
                    # Debug log the raw output
                    llm_logger.debug(f"--- Raw LLM Output (Chunk {i+1}) ---\n{output_text}\n---------------------------")
                    
                    # caution: small model output_txt can make wrong formatted json.
                    sample_ontology = self._parse_json(output_text)
                    
                    # Self-correction logic
                    if not sample_ontology and retry_count > 0:
                        logger.warning(f"Initial discovery JSON parsing failed for chunk {i+1}. Attempting self-correction...")
                        correction_prompt = f"The following JSON was invalid. Please fix the formatting and return ONLY the valid JSON object:\n\n{output_text}"
                        retry_response = self.llm.complete(correction_prompt)
                        sample_ontology = self._parse_json(retry_response.text.strip())

                    if sample_ontology:
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
                        logger.warning(f"No valid JSON block found in chunk {i+1}. Enable --debug to see raw output.")
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
            
            return self._parse_json(output_text)
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
        return raw_ontology

    def _merge_ontologies(self, base: dict, new: dict):
        existing_node_labels = {n['label'].lower(): n for n in base['node_types']}
        for node in new.get('node_types', []):
            if node['label'].lower() not in existing_node_labels:
                base['node_types'].append(node)
                existing_node_labels[node['label'].lower()] = node
        
        # Dictionary to track existing relationships and their indices for easy updating
        existing_rels = {}
        for i, r in enumerate(base['relationship_types']):
            key = (r['source'].lower(), r['type'].lower(), r['target'].lower())
            existing_rels[key] = i

        for rel in new.get('relationship_types', []):
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