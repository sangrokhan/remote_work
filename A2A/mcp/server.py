import asyncio
import pandas as pd
import json
import os
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("FileSystemServer")

# Define base path for security (sandbox)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

@mcp.tool()
def read_text_file(filename: str) -> str:
    """Reads a text file from the data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return f"Error: File {filename} not found in {DATA_DIR}"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
def read_csv_file(filename: str) -> str:
    """Reads a CSV file from the data directory and returns it as a JSON string."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return f"Error: File {filename} not found in {DATA_DIR}"
    try:
        df = pd.read_csv(filepath)
        return df.to_json(orient="records")
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

@mcp.tool()
def write_email_file(filename: str, content: str) -> str:
    """Writes content to a file in the output directory."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

if __name__ == "__main__":
    mcp.run()
