# 3GPP Documentation to Neo4j Graph Pipeline

This project provides a pipeline to chunk documentation, extract knowledge triples using a local LLM, and store them in a Neo4j graph database.

## Prerequisites

- Python 3.9+
- Docker & Docker Compose
- A running LLM server (vLLM, Ollama, etc.) providing an OpenAI-compatible API.

## Setup

1.  **Clone the repository** (if applicable).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Start Neo4j**:
    ```bash
    docker compose up -d
    ```
4.  **Configure Environment**:
    - Copy `.env.example` to `.env`:
      ```bash
      cp .env.example .env
      ```
    - Edit `.env` and set your `LLM_API_BASE` (IP and Port) and other configurations.

## Usage

### 1. Ontology Discovery (Optional)
If you don't have a fixed schema, you can discover one from a document:
```bash
python ontology_discovery.py data/your_spec.md -o ontology.json
```
- Use `--schema schema.json` to enforce specific node labels during discovery.
- Use `--full` for a comprehensive scan.

### 2. Triple Extraction
Run the pipeline to extract and store knowledge triples:
```bash
python graph_pipeline.py data/your_spec.md
```
- **Custom Schema**: You can override the default ontology by providing separate files:
  ```bash
  python graph_pipeline.py data/your_spec.md --node-types node_types.json --relation-types relation_types.json
  ```

The script will:
1.  Read the document from `data/sample.txt`.
2.  Split it into chunks based on `CHUNK_SIZE` and `CHUNK_OVERLAP`.
3.  Send each chunk to the LLM to extract triples (Head, Relationship, Tail).
4.  Assign a unique `chunk_id` to each extracted triple.
5.  Save the entities and relationships into Neo4j.

## Visualizing the Graph

Open your browser and go to:
- **URL**: [http://localhost:7474](http://localhost:7474)
- **Username**: `neo4j`
- **Password**: (as set in `.env`, default is `password`)

You can run the following Cypher query to see your data:
```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100;
```
