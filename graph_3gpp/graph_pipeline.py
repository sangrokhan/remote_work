import os
import json
import logging
import argparse
import sys
from datetime import datetime
from typing import List, Dict, Any
from uuid import uuid4

from neo4j import GraphDatabase
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.llms.openai_like import OpenAILike
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphPipeline:
    def __init__(self, node_types_path="node_types.json", rel_types_path="relation_types.json", knowledge_path=None):
        # Load configurations from environment
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        self.api_base = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
        self.api_key = os.getenv("LLM_API_KEY", "EMPTY")
        self.model_name = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "300.0"))
        
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 256))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 20))
        self.ontology_path = os.getenv("ONTOLOGY_PATH", "ontology.json")
        self.node_types_path = node_types_path
        self.rel_types_path = rel_types_path
        self.knowledge_path = knowledge_path
        self.extraction_prompt = os.getenv("TRIPLE_EXTRACTION_PROMPT", "")

        # Initialize components
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        self.chunker = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        
        # Load Limitations
        self.allowed_node_types = self._load_json_file(self.node_types_path, [])
        self.allowed_rel_types = self._load_json_file(self.rel_types_path, [])
        
        # Load Discovered Metadata (Ontology)
        self.ontology_data = self._load_json_file(self.ontology_path, {})
        self.knowledge_base = self._load_knowledge()
        
        self.llm = OpenAILike(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model_name,
            is_chat_model=True,
            timeout=self.timeout,
            max_tokens=2048
        )

    def _load_json_file(self, path: str, default: Any) -> Any:
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load JSON from {path}: {e}")
        return default

    def _load_knowledge(self) -> str:
        if self.knowledge_path and os.path.exists(self.knowledge_path):
            try:
                with open(self.knowledge_path, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                logger.error(f"Failed to load knowledge from {self.knowledge_path}: {e}")
        return ""

    def _get_relevant_ontology_str(self, text: str) -> str:
        """Filters the ontology to only include elements relevant to the text chunk."""
        if not self.ontology_data:
            return "Generic (Subject, Predicate, Object)"
        
        nodes = self.ontology_data.get("nodes", [])
        relations = self.ontology_data.get("relations", [])
        
        compact_ontology = {
            "nodes": nodes[:50],
            "relations": relations[:100]
        }
        
        return json.dumps(compact_ontology, separators=(',', ':'))

    def close(self):
        self.driver.close()

    def chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Reads a file and splits it into chunks, respecting document structure and size limits."""
        logger.info(f"Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 1. Structural parsing (e.g., by Markdown headers)
        if file_path.lower().endswith(('.md', '.markdown')):
            from llama_index.core.node_parser import MarkdownNodeParser
            structural_parser = MarkdownNodeParser()
            initial_nodes = structural_parser.get_nodes_from_documents([Document(text=text)])
        else:
            initial_nodes = [Document(text=text)]
        
        # 2. Recursive refinement: ensure each structural node fits in the chunk_size
        final_chunks = []
        for node in initial_nodes:
            content = node.get_content().strip()
            if not content:
                continue
            
            # If the node is already small enough, keep it
            if len(content) <= self.chunk_size:
                final_chunks.append({
                    "chunk_id": str(uuid4()), 
                    "text": content,
                    "source": file_path
                })
            else:
                # If too large, split it further using SentenceSplitter
                refined_nodes = self.chunker.get_nodes_from_documents([Document(text=content)])
                for r_node in refined_nodes:
                    r_content = r_node.get_content().strip()
                    if r_content:
                        final_chunks.append({
                            "chunk_id": str(uuid4()), 
                            "text": r_content,
                            "source": file_path
                        })
        
        logger.info(f"Generated {len(final_chunks)} chunks after structural and size-based splitting.")
        return final_chunks

    def _log_parse_failure(self, raw_text: str, error: str, context: str = ""):
        """Logs failed JSON parsing attempts to a dedicated file for debugging."""
        with open("json_parse_fails.log", "a", encoding="utf-8") as f:
            f.write(f"--- FAILURE AT {datetime.now().isoformat()} ---\n")
            f.write(f"CONTEXT: {context}\n")
            f.write(f"ERROR: {error}\n")
            f.write("RAW TEXT:\n")
            f.write(raw_text)
            f.write("\n" + "="*50 + "\n\n")

    def _parse_json(self, text: str, context: str = "") -> Dict[str, Any]:
        """Extracts and parses JSON from LLM response using dirtyjson for robustness."""
        import dirtyjson
        import re
        
        # 1. Find the start and end of the JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            self._log_parse_failure(text, "No JSON object markers found", context)
            return {}
        
        json_str = text[start:end+1]
        
        try:
            return dirtyjson.loads(json_str)
        except Exception as e:
            logger.error(f"JSON Parsing Error with dirtyjson: {e}")
            try:
                # Basic cleanup
                clean_str = re.sub(r'//.*?\n|/\*.*?\*/', '', json_str, flags=re.DOTALL)
                return dirtyjson.loads(clean_str)
            except Exception as e2:
                self._log_parse_failure(json_str, f"Final parsing failed: {e2}", context)
                return {}

    def extract_triples(self, chunk: Dict[str, Any], retry_count: int = 1) -> List[Dict[str, Any]]:
        """Calls LLM to extract triples from text based on ontology with self-correction logic."""
        text = chunk["text"]
        chunk_id = chunk["chunk_id"]

        # Get relevant ontology for this chunk to save tokens
        ontology_str = self._get_relevant_ontology_str(text)
        
        # Combine ontology with knowledge base
        context = f"ONTOLOGY:\n{ontology_str}"
        if self.knowledge_base:
            context += f"\n\nGENERAL KNOWLEDGE & RULES:\n{self.knowledge_base}"
            
        prompt = self.extraction_prompt.format(
            node_types=json.dumps(self.allowed_node_types),
            relation_types=json.dumps(self.allowed_rel_types),
            ontology=ontology_str,
            text=text
        )

        try:
            response = self.llm.complete(prompt)
            output_text = response.text.strip()
            
            data = self._parse_json(output_text, context=f"Chunk: {chunk_id[:8]}")
            
            # Self-correction logic
            if not data and retry_count > 0:
                logger.warning(f"Initial JSON parsing failed for chunk {chunk_id[:8]}. Attempting self-correction...")
                correction_prompt = f"The following JSON was invalid. Please fix the formatting (ensure double quotes, commas, and proper escaping) and return ONLY the valid JSON object:\n\n{output_text}"
                retry_response = self.llm.complete(correction_prompt)
                data = self._parse_json(retry_response.text.strip(), context=f"Chunk: {chunk_id[:8]} (RETRY)")

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
        """Saves triples to Neo4j using ontological labels and properties with additive merging."""
        if not triples:
            return

        with self.driver.session() as session:
            for triple in triples:
                h_label = triple.get('head_label', 'Entity').replace(" ", "_")
                t_label = triple.get('tail_label', 'Entity').replace(" ", "_")
                rel_type = triple.get('type', 'RELATED_TO').upper().replace(" ", "_").replace("-", "_")
                
                # Extract properties
                h_props = triple.get('head_properties', {})
                t_props = triple.get('tail_properties', {})
                edge_props = triple.get('edge_properties', {})
                
                # Standardize names
                h_name = triple.get('head', 'Unknown')
                t_name = triple.get('tail', 'Unknown')
                
                # Metadata
                chunk_id = triple.get('chunk_id')

                # Clean property keys for Neo4j compatibility
                def clean_dict(d):
                    return {"".join(c for c in str(k) if c.isalnum() or c == '_'): v for k, v in d.items() if k}

                h_props_clean = clean_dict(h_props)
                t_props_clean = clean_dict(t_props)
                edge_props_clean = clean_dict(edge_props)
                
                # Cypher logic: 
                # 1. MERGE nodes by name
                # 2. Add properties additively (concatenate if different)
                # 3. Collect chunk_ids in a list
                
                additive_node_query = (
                    f"MERGE (n:`{h_label}` {{name: $name}}) "
                    f"WITH n "
                    f"UNWIND keys($props) AS key "
                    f"SET n[key] = CASE "
                    f"  WHEN n[key] IS NULL THEN $props[key] "
                    f"  WHEN n[key] CONTAINS $props[key] THEN n[key] "
                    f"  ELSE n[key] + ' | ' + $props[key] "
                    f"END "
                    f"SET n.chunk_ids = apoc.coll.toSet(coalesce(n.chunk_ids, []) + $chunk_id)"
                )
                
                # Fallback for chunk_ids if APOC is not available
                simple_node_query = (
                    f"MERGE (n:`{{label}}` {{name: $name}}) "
                    f"WITH n "
                    f"UNWIND keys($props) AS key "
                    f"SET n[key] = CASE "
                    f"  WHEN n[key] IS NULL THEN $props[key] "
                    f"  WHEN n[key] CONTAINS $props[key] THEN n[key] "
                    f"  ELSE n[key] + ' | ' + $props[key] "
                    f"END "
                )

                try:
                    # Save Head
                    session.run(simple_node_query.replace("{label}", h_label), name=h_name, props=h_props_clean)
                    # Save Tail
                    session.run(simple_node_query.replace("{label}", t_label), name=t_name, props=t_props_clean)
                    
                    # Save Relationship
                    rel_query = (
                        f"MATCH (h:`{h_label}` {{name: $h_name}}) "
                        f"MATCH (t:`{t_label}` {{name: $t_name}}) "
                        f"MERGE (h)-[r:`{rel_type}`]->(t) "
                        f"WITH r "
                        f"UNWIND keys($props) AS key "
                        f"SET r[key] = CASE "
                        f"  WHEN r[key] IS NULL THEN $props[key] "
                        f"  WHEN r[key] CONTAINS $props[key] THEN r[key] "
                        f"  ELSE r[key] + ' | ' + $props[key] "
                        f"END "
                    )
                    session.run(rel_query, h_name=h_name, t_name=t_name, props=edge_props_clean)
                    
                except Exception as e:
                    logger.error(f"Failed to save triple {triple}: {e}")

        logger.info(f"Processed {len(triples)} triples in Neo4j.")

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
    parser.add_argument("--node-types", help="Path to JSON file containing allowed node types", default="node_types.json")
    parser.add_argument("--relation-types", help="Path to JSON file containing allowed relationship types", default="relation_types.json")
    parser.add_argument("--knowledge", help="Path to text file containing general domain knowledge", default="knowledge.txt")
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
        
    pipeline = GraphPipeline(
        node_types_path=args.node_types, 
        rel_types_path=args.relation_types,
        knowledge_path=args.knowledge
    )
    pipeline.run(args.file)

if __name__ == "__main__":
    main()
