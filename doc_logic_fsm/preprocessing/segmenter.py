import re
import json
import os

def segment_3gpp_clauses(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Simple segmentation by line and Clause markers
    lines = content.split('\n')
    clauses = []
    
    current_clause = ""
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Check if it starts with digit> (e.g., 1>, 2>)
        if re.match(r'^\d+>', line):
            clauses.append(line)
        elif re.match(r'^\d+(\.\d+)*', line): # Clause header
            clauses.append(line)
        else:
            clauses.append(line)
            
    return clauses

if __name__ == "__main__":
    # Use relative path from script location
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    snippet_path = os.path.join(base_dir, "docs", "TS38331_snippet.txt")
    if os.path.exists(snippet_path):
        result = segment_3gpp_clauses(snippet_path)
        print(json.dumps(result, indent=2))
