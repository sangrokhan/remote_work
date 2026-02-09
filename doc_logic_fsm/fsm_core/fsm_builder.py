import networkx as nx
import json
import os

class FSMBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_transition(self, from_state, to_state, trigger, actions):
        label = f"Trigger: {trigger}\nActions: {', '.join(actions)}"
        self.graph.add_edge(from_state, to_state, trigger=trigger, actions=actions, label=label)

    def build_from_extracted_logic(self, logic_data, initial_state="RRC_IDLE"):
        current_trigger = None
        current_actions = []
        
        for item in logic_data:
            logic = item["logic"]
            if logic["triggers"]:
                current_trigger = logic["triggers"][0]
            
            if logic["actions"]:
                current_actions.extend(logic["actions"])
                
            if logic["state_transitions"]:
                target_state = logic["state_transitions"][0]
                if current_trigger:
                    self.add_transition(initial_state, target_state, current_trigger, current_actions)
        
        return self.graph

    def save_graph_json(self, path):
        data = nx.node_link_data(self.graph)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    extracted_logic = [
      {
        "text": "The UE shall perform the following actions upon reception of the RRCSetup:",
        "logic": {"triggers": ["RRCSetup"], "actions": [], "state_transitions": []}
      },
      {
        "text": "1> stop timer T300, if running;",
        "logic": {"triggers": [], "actions": ["STOP_T300"], "state_transitions": []}
      },
      {
        "text": "2> enter RRC_CONNECTED state;",
        "logic": {"triggers": [], "actions": [], "state_transitions": ["RRC_CONNECTED"]}
      },
      {
        "text": "2> submit the RRCSetupComplete message",
        "logic": {"triggers": [], "actions": ["SEND_RRCSetupComplete"], "state_transitions": []}
      }
    ]
    
    builder = FSMBuilder()
    builder.build_from_extracted_logic(extracted_logic)
    
    # Use relative path from script location
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    output_path = os.path.join(base_dir, "fsm_core", "rrc_fsm.json")
    builder.save_graph_json(output_path)
    print(f"FSM logic saved to {output_path}")
