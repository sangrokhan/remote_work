import os
import json
import logging
import re
import argparse
import sys
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

    def discover(self, file_path: str) -> dict:
        """Analyzes a document to discover its underlying ontology."""
        logger.info(f"Analyzing {file_path} for ontology discovery...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Use a representative sample for schema induction
            sample_text = text[:4000]

            prompt_template = ONTOLOGY_DISCOVERY_PROMPT.replace("\\n", "\n")
            prompt = prompt_template.format(text=sample_text)

            response = self.llm.complete(prompt)
            output_text = response.text.strip()
            logger.info(f"Raw LLM Output received.")
            
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

    def save_ontology(self, ontology: dict, output_path: str):
        """Saves the discovered ontology to a file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ontology, f, indent=2)
        logger.info(f"Ontology saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="3GPP Ontology Discovery Tool")
    parser.add_argument("file", help="Path to the document to analyze")
    parser.add_argument("-o", "--output", help="Path to save the generated ontology", default=DEFAULT_ONTOLOGY_PATH)
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
        
    discovery = OntologyDiscovery()
    result = discovery.discover(args.file)
    
    if result:
        discovery.save_ontology(result, args.output)
        print(f"Successfully discovered and saved ontology to {args.output}")
    else:
        logger.error("Failed to discover ontology.")
        sys.exit(1)

if __name__ == "__main__":
    main()
