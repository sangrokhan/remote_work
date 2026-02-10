import os
import json
import logging
import argparse
import sys
from typing import List, Dict, Any
from uuid import uuid4

from neo4j import GraphDatabase
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.llms.openai_like import OpenAILike
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphPipeline:
    def __init__(self):
        # Load configurations from environment
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        self.api_base = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
        self.api_key = os.getenv("LLM_API_KEY", "EMPTY")
        self.model_name = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "60.0"))
        
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 256))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 20))
        self.ontology_path = os.getenv("ONTOLOGY_PATH", "ontology.json")
        self.extraction_prompt = os.getenv(
            "TRIPLE_EXTRACTION_PROMPT",
            "Extract triples from the text.\nONTOLOGY:\n{ontology}\nText:\n{text}\nJSON Output:"
        ).replace("\\n", "\n")

        # Initialize components
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        self.chunker = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.ontology_data = self._load_ontology()
        
        self.llm = OpenAILike(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model_name,
            is_chat_model=True,
            timeout=self.timeout
        )

    def _load_ontology(self) -> Dict[str, Any]:
        if os.path.exists(self.ontology_path):
            try:
                with open(self.ontology_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load ontology JSON: {e}")
        logger.warning(f"Ontology file {self.ontology_path} not found or invalid. Using empty schema.")
        return {}

    def _get_relevant_ontology_str(self, text: str) -> str:
        """Filters the ontology to only include elements relevant to the text chunk."""
        if not self.ontology_data:
            return "Generic (Subject, Predicate, Object)"
        
        import re
        node_types = self.ontology_data.get("node_types", [])
        rel_types = self.ontology_data.get("relationship_types", [])
        
        # 1. Identify relevant node labels by checking their presence in text
        relevant_labels = set()
        # Heuristic: search for node labels in the text chunk
        # To avoid massive regex, we only check if the label is mentioned
        text_lower = text.lower()
        for node in node_types:
            label = node.get("label", "")
            if label.lower() in text_lower:
                relevant_labels.add(label)
        
        # 2. Filter node types and strip descriptions to save tokens
        filtered_nodes = [
            {"label": n["label"]}
            for n in node_types if n["label"] in relevant_labels
        ]
        
        # 3. Filter relationship types
        # Include relationships where either source or target is in relevant_labels
        filtered_rels = [
            {"source": r["source"], "type": r["type"], "target": r["target"]}
            for r in rel_types 
            if r["source"] in relevant_labels or r["target"] in relevant_labels
        ]
        
        # 4. Token management: If still too large or nothing found
        # If nothing found, provide a subset of the ontology as a hint
        if not filtered_nodes:
            filtered_nodes = [{"label": n["label"]} for n in node_types[:20]]
            filtered_rels = [
                {"source": r["source"], "type": r["type"], "target": r["target"]}
                for r in rel_types[:20]
            ]
        
        # Limit to reasonable number of types to stay under token limits
        filtered_nodes = filtered_nodes[:100]
        filtered_rels = filtered_rels[:150]

        compact_ontology = {
            "node_types": filtered_nodes,
            "relationship_types": filtered_rels
        }
        
        return json.dumps(compact_ontology, separators=(',', ':'))

    def close(self):
        self.driver.close()

    def chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Reads a file and splits it into chunks."""
        logger.info(f"Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        doc = Document(text=text)
        nodes = self.chunker.get_nodes_from_documents([doc])
        
        chunks = []
        for node in nodes:
            chunks.append({
                "chunk_id": str(uuid4()), 
                "text": node.get_content(),
                "source": file_path
            })
        
        logger.info(f"Generated {len(chunks)} chunks.")
        return chunks

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extracts and parses JSON from LLM response, with basic cleaning."""
        import re
        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return {}
        
        json_str = text[start:end+1]
        
        # Basic cleaning to handle common LLM issues in technical docs
        # 1. Handle common unescaped backslashes (e.g., in technical markers or paths)
        # Only escape backslashes that are NOT part of a valid JSON escape sequence
        json_str = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', json_str)
        
        # 2. Try to fix missing commas between fields
        # Between string and string: "val" "key"
        json_str = re.sub(r'("(?:\\["\\/bfnrtu]|[^"\\])*")\s+(")', r'\1, \2', json_str)
        # Between brace/bracket and string: } "key" or ] "key"
        json_str = re.sub(r'([}\]])\s+(")', r'\1, \2', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error: {e}")
            logger.debug(f"Failed JSON string: {json_str}")
            return {}

    def extract_triples(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calls LLM to extract triples from text based on ontology."""
        text = chunk["text"]
        chunk_id = chunk["chunk_id"]

        # Get relevant ontology for this chunk to save tokens
        ontology_str = self._get_relevant_ontology_str(text)
        prompt = self.extraction_prompt.format(ontology=ontology_str, text=text)

        try:
            response = self.llm.complete(prompt)
            output_text = response.text.strip()
            
            data = self._parse_json(output_text)
            if not data:
                logger.error(f"Failed to extract valid JSON for chunk {chunk_id}.")
                return []

            triples = data.get("triples", [])
            
            for triple in triples:
                triple["chunk_id"] = chunk_id
                
            logger.info(f"Extracted {len(triples)} triples from chunk {chunk_id[:8]}...")
            return triples

        except Exception as e:
            logger.error(f"Error during triple extraction: {e}")
            return []

    def save_to_neo4j(self, triples: List[Dict[str, Any]]):
        """Saves triples to Neo4j using ontological labels."""
        if not triples:
            return

        with self.driver.session() as session:
            for triple in triples:
                h_label = triple.get('head_label', 'Entity').replace(" ", "_")
                t_label = triple.get('tail_label', 'Entity').replace(" ", "_")
                rel_type = triple.get('type', 'RELATED_TO').upper().replace(" ", "_").replace("-", "_")
                
                dynamic_query = (
                    f"MERGE (h:`{h_label}` {{name: $head}}) "
                    f"MERGE (t:`{t_label}` {{name: $tail}}) "
                    f"MERGE (h)-[r:`{rel_type}`]->(t) "
                    "SET r.chunk_id = $chunk_id"
                )
                
                try:
                    session.run(dynamic_query, head=triple['head'], tail=triple['tail'], chunk_id=triple['chunk_id'])
                except Exception as e:
                    logger.error(f"Failed to save triple {triple}: {e}")

        logger.info(f"Saved {len(triples)} triples to Neo4j.")

    def run(self, file_path: str):
        try:
            chunks = self.chunk_document(file_path)
            
            triples_found = False
            for chunk in chunks:
                triples = self.extract_triples(chunk)
                if triples:
                    self.save_to_neo4j(triples)
                    triples_found = True
                
            if triples_found:
                logger.info("Pipeline completed successfully.")
            else:
                logger.warning("No triples were extracted.")
                
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
        finally:
            self.close()

def main():
    parser = argparse.ArgumentParser(description="3GPP Graph Pipeline")
    parser.add_argument("file", help="Path to the document to process")
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
        
    pipeline = GraphPipeline()
    pipeline.run(args.file)

if __name__ == "__main__":
    main()
