import io
import os
import tempfile

import yaml
import pandas as pd
import xlwings as xw
from decrypter import decrypt_excel
from graph_client import Neo4jClient


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_excel_path(decrypted_file):
    if isinstance(decrypted_file, (str, os.PathLike)):
        return str(decrypted_file), False

    if isinstance(decrypted_file, (bytes, bytearray, memoryview)):
        data = bytes(decrypted_file)
    elif isinstance(decrypted_file, io.BytesIO):
        data = decrypted_file.getvalue()
    elif hasattr(decrypted_file, "getvalue"):
        data = decrypted_file.getvalue()
    else:
        raise TypeError(f"Unsupported decrypted file type: {type(decrypted_file)}")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    temp.write(data)
    temp.close()
    return temp.name, True


def _read_excel_sheet_to_df(path):
    app = xw.App(visible=False, add_book=False)
    app.display_alerts = False
    app.screen_updating = False

    workbook = None
    try:
        workbook = app.books.open(path)
        worksheet = workbook.sheets[0]
        raw_data = worksheet.used_range.options(ndim=2).value
    finally:
        if workbook is not None:
            workbook.close()
        app.quit()

    if not raw_data:
        return pd.DataFrame()

    header = list(raw_data[0] or [])
    rows = raw_data[1:] if len(raw_data) > 1 else []
    return pd.DataFrame(rows or [], columns=header)


def process_file(client, file_config):
    path = file_config['path']
    doc_name = file_config['name']
    key_col = file_config['key_column']
    password = file_config.get("password")

    print(f"Processing {doc_name} from {path}...")

    # Decrypt
    decrypted_file = decrypt_excel(path, password=password)
    temp_path = None
    use_temp = False
    try:
        temp_path, use_temp = _resolve_excel_path(decrypted_file)
        # Load into Pandas via xlwings
        df = _read_excel_sheet_to_df(temp_path)
    finally:
        if use_temp and temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

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
        for file_cfg in config['files']:
            process_file(client, file_cfg)
    finally:
        client.close()

if __name__ == "__main__":
    main()
