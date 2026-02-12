import pandas as pd
import yaml
import os
from decrypter import decrypt_excel
from graph_client import Neo4jClient

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_file(client, file_config, password):
    path = file_config['path']
    doc_name = file_config['name']
    key_col = file_config['key_column']

    print(f"Processing {doc_name} from {path}...")
    
    # Decrypt
    decrypted_file = decrypt_excel(path, password)
    
    # Load into Pandas
    df = pd.read_excel(decrypted_file, engine='openpyxl')
    
    # Ingest into Neo4j
    client.ingest_data(doc_name, df, key_col)
    print(f"Successfully ingested {len(df)} rows from {doc_name}.")

def main():
    if not os.path.exists("config.yaml"):
        print("Error: config.yaml not found.")
        return

    config = load_config()
    
    # Initialize Neo4j Client
    neo4j_conf = config['neo4j']
    client = Neo4jClient(neo4j_conf['uri'], neo4j_conf['user'], neo4j_conf['password'])
    
    try:
        password = config.get('excel_password')
        for file_cfg in config['files']:
            process_file(client, file_cfg, password)
    finally:
        client.close()

if __name__ == "__main__":
    main()
