import json
import os

def generate_interaction_mermaid(logic_json_path):
    with open(logic_json_path, 'r') as f:
        data = json.load(f)
    
    mermaid_str = "sequenceDiagram\n    participant RRC\n    participant MAC\n    participant PHY\n\n"
    mermaid_str += "    Note over RRC: Connection Start\n"
    
    for entry in data.get("TS38321_snippet.txt", []):
        for logic in entry["logic"]:
            if logic["type"] == "INTER_LAYER_TRIGGER":
                mermaid_str += f"    RRC->>MAC: Trigger {logic['data']['proc']}\n"
            if logic["type"] == "CROSS_LAYER_ACT":
                if "physical layer" in logic["data"]["entity"]:
                    mermaid_str += f"    MAC->>PHY: {logic['data']['act']}\n"
            if logic["type"] == "INTER_LAYER_IND":
                if "upper layers" in logic["data"]["target"]:
                    mermaid_str += f"    MAC->>RRC: Indicate {logic['data']['evt']}\n"

    return mermaid_str

def generate_interaction_html(mermaid_str, output_path):
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{ startOnLoad: true }});</script>
</head>
<body>
    <h1>Dynamic Interaction Sequence</h1>
    <div class="mermaid">
{mermaid_str}
    </div>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_template)

if __name__ == "__main__":
    # Use relative path from script location
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    logic_path = os.path.join(base_dir, "fsm_core", "multi_layer_logic.json")
    if os.path.exists(logic_path):
        m_str = generate_interaction_mermaid(logic_path)
        print(m_str)
        output_mmd = os.path.join(base_dir, "validation", "interaction_diagram.mmd")
        with open(output_mmd, "w") as f:
            f.write(m_str)
            
        output_html = os.path.join(base_dir, "validation", "interaction_viewer.html")
        generate_interaction_html(m_str, output_html)
        print(f"Updated dynamic HTML at {output_html}")
