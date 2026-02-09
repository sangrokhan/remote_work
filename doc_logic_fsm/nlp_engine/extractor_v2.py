import re
import json
import os

class AdvancedLogicExtractor:
    def __init__(self):
        self.trigger_pattern = r"upon reception of (the\s+)?(?P<msg>[\w-]+)"
        self.condition_pattern = r"if (the\s+)?(?P<cond>.+?):"
        self.action_patterns = {
            "STOP_TIMER": r"stop timer (?P<timer>T\d+)",
            "ENTER_STATE": r"enter (?P<state>[\w_]+) state",
            "SEND_MSG": r"submit (the\s+)?(?P<msg>[\w-]+) message",
            "PROC_END": r"the procedure ends"
        }

    def extract_from_lines(self, lines):
        transitions = []
        current_trigger = None
        current_conditions = []
        
        # This is a simplified stateful parser for the hierarchical 3GPP format
        for line in lines:
            line = line.strip()
            
            # 1. Trigger
            t_match = re.search(self.trigger_pattern, line, re.IGNORECASE)
            if t_match:
                current_trigger = t_match.group("msg")
                continue

            # 2. Condition (Hierarchical)
            c_match = re.search(self.condition_pattern, line, re.IGNORECASE)
            if c_match:
                # For simplicity, we just keep track of the most recent 'if'
                current_conditions.append(c_match.group("cond"))
                continue

            # 3. Actions & State Transitions
            found_action = False
            for act_type, pattern in self.action_patterns.items():
                a_match = re.search(pattern, line, re.IGNORECASE)
                if a_match:
                    found_action = True
                    action_val = a_match.group(1) if len(a_match.groups()) > 0 else act_type
                    
                    if act_type == "ENTER_STATE":
                        transitions.append({
                            "from": "WAITING_FOR_SETUP",
                            "to": a_match.group("state"),
                            "trigger": current_trigger,
                            "condition": " AND ".join(current_conditions),
                            "actions": [] # Will fill later if needed
                        })
                    else:
                        # Add action to the last transition or keep as global action
                        if transitions:
                            transitions[-1]["actions"].append(f"{act_type}({action_val})")
        
        return transitions

if __name__ == "__main__":
    # Use relative path from script location
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    input_path = os.path.join(base_dir, "docs", "TS38331_snippet.txt")
    
    if os.path.exists(input_path):
        with open(input_path, "r") as f:
            lines = f.readlines()
        
        extractor = AdvancedLogicExtractor()
        logic = extractor.extract_from_lines(lines)
        
        output_path = os.path.join(base_dir, "fsm_core", "nr_rrc_logic.json")
        with open(output_path, 'w') as f:
            json.dump(logic, f, indent=2)
        print(f"Advanced logic extracted to {output_path}")
