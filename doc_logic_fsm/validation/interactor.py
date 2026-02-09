import json

def generate_interaction_mermaid(logic_json_path):
    with open(logic_json_path, 'r') as f:
        data = json.load(f)
    
    mermaid_str = "sequenceDiagram\n    participant RRC\n    participant MAC\n    participant PHY\n\n"
    
    # Process RRC (TS 38.331)
    # Simple heuristic to show RRC triggering MAC
    mermaid_str += "    Note over RRC: Connection Start\n"
    
    # Process MAC (TS 38.321)
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

if __name__ == "__main__":
    logic_path = "/home/han/repo/remote_work/doc_logic_fsm/fsm_core/multi_layer_logic.json"
    m_str = generate_interaction_mermaid(logic_path)
    print(m_str)
    with open("/home/han/repo/remote_work/doc_logic_fsm/validation/interaction_diagram.mmd", "w") as f:
        f.write(m_str)
