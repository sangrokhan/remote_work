import json
import os

def generate_mermaid(fsm_json_path):
    with open(fsm_json_path, 'r') as f:
        data = json.load(f)
    
    mermaid_str = "stateDiagram-v2\n"
    for link in data['edges']:
        source = link['source']
        target = link['target']
        trigger = link.get('trigger', '')
        actions = link.get('actions', [])
        
        action_str = "\\n".join(actions)
        label = f"{trigger} / {action_str}"
        mermaid_str += f"    {source} --> {target}: {label}\n"
    
    return mermaid_str

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fsm_path = os.path.join(base_dir, "fsm_core/rrc_fsm.json")
    if os.path.exists(fsm_path):
        m_str = generate_mermaid(fsm_path)
        print(m_str)
        
        output_path = os.path.join(base_dir, "validation/fsm_diagram.mmd")
        with open(output_path, "w") as f:
            f.write(m_str)
