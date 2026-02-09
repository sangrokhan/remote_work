import re
import json
import os
import networkx as nx

class AutoFSMExtractor:
    def __init__(self, md_path):
        self.md_path = md_path
        self.content = ""
        self.states = set()
        self.transitions = []
        self.blacklist = {
            "3GPP", "TS", "UE", "NR", "RAN", "SIB", "MAC", "PDCP", "RLC", "SRB", "DRB", "LTE", "CELL", "NETWORK",
            "PROCEDURE", "PROCEDURES", "MESSAGE", "MESSAGES", "VALUE", "VALUES", "IES", "FIELDS", "INFORMATION",
            "CONTENTS", "CONFIGURATION", "CONDITION", "CONDITIONS", "START", "STOP", "RESULT", "ACCESS", "FAILURE",
            "FREQUENCY", "BAND", "PCEL", "SCG", "MCG", "DATA", "SUCCESS", "TRIGGER", "TRIGGERED", "INDICATION", "FALLBACK"
        }

    def load_document(self):
        if not os.path.exists(self.md_path):
            print(f"[Error] File not found: {self.md_path}")
            return False
        with open(self.md_path, 'r', encoding='utf-8') as f:
            # Clean up some common markdown artifacts that break regex
            self.content = f.read().replace("\\_", "_").replace("*", "")
        return True

    def discover_states(self):
        """
        Automatically discover RRC states by looking for common patterns:
        1. Explicit state definitions (e.g., "**- RRC_IDLE**")
        2. Words ending in _IDLE, _CONNECTED, _INACTIVE
        3. Explicit state machine descriptions
        """
        print("[*] Automatically discovering states from content...")
        
        # Pattern 1: Definitions in Section 4.2.1
        # We look for the start of section 4.2.1 and scan until the next major section
        section_match = re.search(r"### 4.2.1 UE states and state transitions.*?###", self.content, re.DOTALL | re.IGNORECASE)
        if section_match:
            section_text = section_match.group(0)
            # Find bullet points with capitalized names
            found = re.findall(r"(?:^|\n)[-\s]+(RRC_[A-Z]+)", section_text)
            self.states.update(found)

        # Pattern 2: Global scan for RRC_ prefixed names
        rrc_states = re.findall(r"(RRC_[A-Z_]+)", self.content)
        self.states.update(rrc_states)

        # Clean up: Remove noise and items in blacklist
        self.states = {s.strip("_") for s in self.states if s not in self.blacklist and len(s) > 5}
        
        # Ensure we at least have the core 3 NR RRC states if they exist in text
        core_states = ["RRC_IDLE", "RRC_CONNECTED", "RRC_INACTIVE"]
        for cs in core_states:
            if cs in self.content:
                self.states.add(cs)

        print(f"[*] Discovered States: {self.states}")

    def extract_transitions(self):
        """
        Find transitions by scanning for sentences that mention multiple states
        or transition keywords near a state name.
        """
        print("[*] Extracting transitions between states...")
        
        # Split content into sentences/bullets for finer analysis
        lines = re.split(r"[\n\.;]", self.content)
        
        transition_verbs = ["transition", "enter", "move", "resume", "suspend", "leave", "go to"]
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            line_upper = line.upper()
            found_states = [s for s in self.states if s in line_upper]
            
            # Case 1: "State A to State B transition" or "from State A to State B"
            for s1 in self.states:
                for s2 in self.states:
                    if s1 == s2: continue
                    # Pattern: "S1 to S2" or "from S1 to S2"
                    if re.search(rf"{s1}.*?to.*?{s2}", line_upper):
                        self.transitions.append({
                            "from": s1,
                            "to": s2,
                            "context": line[:100] + ("..." if len(line) > 100 else "")
                        })
                    # Pattern: "S2 from S1" (e.g., transition to S2 from S1)
                    elif re.search(rf"{s2}.*?from.*?{s1}", line_upper):
                        self.transitions.append({
                            "from": s1,
                            "to": s2,
                            "context": line[:100] + ("..." if len(line) > 100 else "")
                        })

            # Case 2: "transition to State B" (Inferring Source from proximity or procedure)
            # For simplicity in this auto-extractor, if we find "enter State B" and only one state is mentioned,
            # we look if the UE was "in State A" recently. 
            # But let's stick to explicit line-based transitions first.
            
            if len(found_states) == 1:
                target = found_states[0]
                if any(verb.upper() in line_upper for verb in transition_verbs):
                    # Check if line indicates "transition TO"
                    if re.search(rf"(?:transition|enter|move|go)\s+(?:to\s+)?{target}", line, re.IGNORECASE):
                        # Try to find a "from" state in the same line or context
                        # If not found, we label it as 'ANY' or 'UNKNOWN' for now
                        self.transitions.append({
                            "from": "OTHER/ANY",
                            "to": target,
                            "context": line[:100]
                        })

    def save_json(self, path):
        # Create a directed graph to deduplicate and structure
        G = nx.MultiDiGraph()
        for state in self.states:
            G.add_node(state)
        
        for t in self.transitions:
            G.add_edge(t['from'], t['to'], context=t['context'])
            
        data = nx.node_link_data(G)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"[*] FSM Data saved to {path}")

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    md_path = os.path.join(base_dir, "docs", "md", "38331-j10.md")
    output_json = os.path.join(base_dir, "fsm_core", "rrc_fsm.json")
    
    extractor = AutoFSMExtractor(md_path)
    if extractor.load_document():
        extractor.discover_states()
        extractor.extract_transitions()
        extractor.save_json(output_json)
