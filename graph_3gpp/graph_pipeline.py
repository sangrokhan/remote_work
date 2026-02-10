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
        
        # Initialize LLM
        self.llm = OpenAILike(
            api_base=LLM_API_BASE,
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
            is_chat_model=True,
            timeout=60.0
        )

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
        """Calls LLM to extract triples from text."""
        text = chunk["text"]
        chunk_id = chunk["chunk_id"]

        prompt = (
            "You are a knowledge graph expert. Extract knowledge triples (Subject, Predicate, Object) "
            "from the following text. 
"
            "Return strictly a JSON object with a key 'triples' containing a list of objects. "
            "Each object must have 'head', 'type', and 'tail'.
"
            "Example format: {"triples": [{"head": "Alice", "type": "KNOWS", "tail": "Bob"}]}

"
            f"Text:
{text}

"
            "JSON Output:"
        )

        try:
            response = self.llm.complete(prompt)
            output_text = response.text.strip()
            
            # Basic cleanup if the LLM includes code blocks
            if output_text.startswith("```json"):
                output_text = output_text[7:]
            if output_text.endswith("```"):
                output_text = output_text[:-3]
                
            data = json.loads(output_text)
            triples = data.get("triples", [])
            
            # Enrich triples with chunk_id
            for triple in triples:
                triple["chunk_id"] = chunk_id
                
            logger.info(f"Extracted {len(triples)} triples from chunk {chunk_id[:8]}...")
            return triples

        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM response for chunk {chunk_id}.")
            logger.debug(f"LLM Output: {output_text}")
            return []
        except Exception as e:
            logger.error(f"Error during triple extraction: {e}")
            return []

    def save_to_neo4j(self, triples: List[Dict[str, Any]]):
        """Saves triples to Neo4j."""
        if not triples:
            return

        query = """
        UNWIND $triples AS t
        MERGE (h:Entity {name: t.head})
        MERGE (tail:Entity {name: t.tail})
        MERGE (h)-[r:RELATIONSHIP {type: t.type}]->(tail)
        SET r.chunk_id = t.chunk_id
        """
        
        # Note: In a real dynamic graph, you'd likely map t.type to the actual Relationship Type dynamically 
        # using APOC or string concatenation (sanitized), because Cypher doesn't allow dynamic types in MERGE 
        # without APOC.
        # For this simplified version, we use a generic :RELATIONSHIP type and store the actual type as a property,
        # OR we can use string formatting if we trust the LLM output (risky but allows visual graph types).
        
        # Better Approach with APOC (if available) or Python-side dynamic query construction.
        # Let's do Python-side construction for types to make it look nice in Neo4j Bloom/Browser.
        
        with self.driver.session() as session:
            for triple in triples:
                # Sanitize type to avoid Cypher injection and syntax errors
                rel_type = triple['type'].upper().replace(" ", "_").replace("-", "_")
                if not rel_type: 
                    rel_type = "RELATED_TO"
                    
                dynamic_query = (
                    "MERGE (h:Entity {name: $head}) "
                    "MERGE (t:Entity {name: $tail}) "
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
            
            all_triples = []
            for chunk in chunks:
                triples = self.extract_triples(chunk)
                all_triples.extend(triples)
                
            if all_triples:
                self.save_to_neo4j(all_triples)
                logger.info("Pipeline completed successfully.")
            else:
                logger.warning("No triples were extracted.")
                
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
        finally:
            self.close()

if __name__ == "__main__":
    pipeline = GraphPipeline()
    # Ensure data/sample.txt exists
    sample_file = "data/sample.txt"
    if os.path.exists(sample_file):
        pipeline.run(sample_file)
    else:
        logger.error(f"File {sample_file} not found.")
