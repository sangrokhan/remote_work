import os
import json
import logging
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

# Configuration from .env
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# LLM Configuration
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 256))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 20))

class GraphPipeline:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.chunker = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        # Load Ontology
        self.ontology = self._load_ontology()
        
        # Initialize LLM
        self.llm = OpenAILike(
            api_base=LLM_API_BASE,
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
            is_chat_model=True,
            timeout=60.0
        )

    def _load_ontology(self) -> str:
        ontology_path = "ontology.json"
        if os.path.exists(ontology_path):
            with open(ontology_path, 'r') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        return "Generic (Subject, Predicate, Object)"

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
                "chunk_id": str(uuid4()), # Generating a unique ID for the chunk
                "text": node.get_content(),
                "source": file_path
            })
        
        logger.info(f"Generated {len(chunks)} chunks.")
        return chunks

    def extract_triples(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calls LLM to extract triples from text based on ontology."""
        text = chunk["text"]
        chunk_id = chunk["chunk_id"]

        prompt = (
            "You are a 3GPP Knowledge Graph expert. Extract knowledge triples from the text "
            "strictly following the provided ONTOLOGY.\n\n"
            "ONTOLOGY:\n"
            f"{self.ontology}\n\n"
            "Instructions:\n"
            "1. Only use Node Labels and Relationship Types defined in the ONTOLOGY.\n"
            "2. If a relationship is not explicitly in the ontology but is critical, use the most similar type.\n"
            "3. Return strictly a JSON object with a key 'triples'.\n"
            "4. Each triple must have: 'head' (node name), 'head_label', 'type' (relationship), 'tail' (node name), 'tail_label'.\n\n"
            f"Text:\n{text}\n\n"
            "JSON Output:"
        )

        try:
            response = self.llm.complete(prompt)
            output_text = response.text.strip()
            
            # More robust JSON extraction
            import re
            json_match = re.search(r"(\{.*\})", output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = output_text

            data = json.loads(json_str)
            triples = data.get("triples", [])
            
            # Enrich triples with chunk_id
            for triple in triples:
                triple["chunk_id"] = chunk_id
                
            logger.info(f"Extracted {len(triples)} triples from chunk {chunk_id[:8]}...")
            return triples

        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM response for chunk {chunk_id}.")
            logger.error(f"LLM Output: {output_text}")
            return []
        except Exception as e:
            logger.error(f"Error during triple extraction: {e}")
            return []

    def save_to_neo4j(self, triples: List[Dict[str, Any]]):
        """Saves triples to Neo4j using ontological labels."""
        if not triples:
            return

        with self.driver.session() as session:
            for triple in triples:
                # Sanitize labels and types
                h_label = triple.get('head_label', 'Entity').replace(" ", "_")
                t_label = triple.get('tail_label', 'Entity').replace(" ", "_")
                rel_type = triple.get('type', 'RELATED_TO').upper().replace(" ", "_").replace("-", "_")
                
                # Cypher query with dynamic labels (using string formatting for labels as they are not parameters)
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

if __name__ == "__main__":
    pipeline = GraphPipeline()
    # Ensure sample_3gpp.txt exists
    sample_file = "sample_3gpp.txt"
    if os.path.exists(sample_file):
        pipeline.run(sample_file)
    else:
        logger.error(f"File {sample_file} not found.")
