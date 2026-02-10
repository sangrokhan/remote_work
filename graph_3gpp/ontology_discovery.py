import os
import json
import logging
import re
from llama_index.llms.openai_like import OpenAILike
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OntologyDiscovery:
    def __init__(self):
        self.llm = OpenAILike(
            api_base=os.getenv("LLM_API_BASE", "http://localhost:8000/v1"),
            api_key=os.getenv("LLM_API_KEY", "EMPTY"),
            model=os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
            is_chat_model=True,
            timeout=120.0
        )

    def discover(self, file_path: str) -> dict:
        """Analyzes a document to discover its underlying ontology."""
        logger.info(f"Analyzing {file_path} for ontology discovery...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Use the first 4000 characters as a representative sample for schema induction
            sample_text = text[:4000]

            prompt = f"""You are an expert in 3GPP standards and Knowledge Graph engineering. 
Analyze the following technical specification and extract the core ONTOLOGY (schema).

Instructions:
1. Identify the key ENTITY TYPES (Nodes) mentioned (e.g., NetworkNode, ProtocolMessage, Timer, Procedure, UserState).
2. Identify the VALID RELATIONSHIPS between these types (e.g., SENDS, TRIGGERED_BY, EXPIRES_DURING, TRANSITIONS_TO).
3. Focus on creating a schema that supports ROOT CAUSE ANALYSIS and SYSTEM OPTIMIZATION.

Return STRICTLY a JSON object with the following structure:
{{
  "node_types": [{{"label": "LabelName", "description": "Brief definition"}}],
  "relationship_types": [{{"source": "SourceLabel", "type": "REL_TYPE", "target": "TargetLabel"}}]
}}

Source Text:
{sample_text}

JSON Output:"""

            response = self.llm.complete(prompt)
            output_text = response.text.strip()
            logger.info(f"Raw LLM Output:\n{output_text}")
            
            # Extract JSON from potential markdown code blocks
            json_match = re.search(r"(\{.*\})", output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = output_text

            ontology = json.loads(json_str)
            return ontology

        except Exception as e:
            logger.error(f"Ontology discovery failed: {e}")
            return {}

    def save_ontology(self, ontology: dict, output_path: str = "ontology.json"):
        """Saves the discovered ontology to a file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, indent=2)
        logger.info(f"Ontology saved to {output_path}")

if __name__ == "__main__":
    discovery = OntologyDiscovery()
    
    # Use the sample file we created
    target_file = "sample_3gpp.txt"
    if os.path.exists(target_file):
        result = discovery.discover(target_file)
        if result:
            discovery.save_ontology(result)
            print(json.dumps(result, indent=2))
        else:
            print("Failed to discover ontology.")
    else:
        print(f"File {target_file} not found. Please provide a valid document path.")