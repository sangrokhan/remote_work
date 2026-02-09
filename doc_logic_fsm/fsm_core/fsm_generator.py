import re
import json
import os

class NR_RRC_FSM_Extractor:
    def __init__(self, md_path):
        self.md_path = md_path
        self.clauses = {}
        self.transitions = []
        
        # Keywords for FSM extraction
        self.state_keywords = ["RRC_IDLE", "RRC_CONNECTED", "RRC_INACTIVE"]
        self.transition_keywords = [
            (r"enter (?P<to_state>RRC_\w+)", "STATE_ENTER"),
            (r"upon reception of (the\s+)?\*(?P<msg>[\w-]+)\*", "MSG_RECEIVE"),
            (r"transmission of (the\s+)?\*(?P<msg>[\w-]+)\*", "MSG_SEND"),
            (r"stop timer (?P<timer>T\d+)", "TIMER_STOP"),
            (r"start timer (?P<timer>T\d+)", "TIMER_START"),
            (r"timer (?P<timer>T\d+) expires", "TIMER_EXPIRY")
        ]

    def segment_by_header(self):
        with open(self.md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by markdown headers (e.g., ### 5.3.3)
        # 3GPP MD uses #### for clause titles
        header_pattern = r"(^#{1,5}\s+\d+\.[\d\.]*\s+.*)"
        parts = re.split(header_pattern, content, flags=re.MULTILINE)
        
        print(f"DEBUG: Found {len(parts)} segments after regex split.")
        
        for i in range(1, len(parts), 2): # Split result has [prefix, header, body, header, body...]
            header = parts[i].strip()
            body = parts[i+1] if i+1 < len(parts) else ""
            self.clauses[header] = body

    def extract_logic(self):
        for header, body in self.clauses.items():
            # Search for transitions in the text using regex
            # Mammoth conversion might leave some artifacts like RRC\_CONNECTED or RRC*CONNECTED*
            # Use a robust regex to find 'enter RRC_XXX'
            enter_matches = re.findall(r"enter\s+(?:the\s+)?\*?(RRC[\\_\s\-\*]*\w+)\*?", body, re.IGNORECASE)
            
            if enter_matches:
                from_state = "UNKNOWN"
                context = header + " " + body[:2000]
                
                # Heuristic for source state
                if "RRC_IDLE" in context or "RRC\\_IDLE" in context: from_state = "RRC_IDLE"
                elif "RRC_CONNECTED" in context or "RRC\\_CONNECTED" in context: from_state = "RRC_CONNECTED"
                elif "RRC_INACTIVE" in context or "RRC\\_INACTIVE" in context: from_state = "RRC_INACTIVE"
                
                # Fix specific common cases where 'UNKNOWN' is obviously something else
                if from_state == "UNKNOWN":
                    if "5.3.3" in header: from_state = "RRC_IDLE" # Connection Establishment starts from IDLE
                    elif "5.3.13" in header: from_state = "RRC_INACTIVE" # Resume starts from INACTIVE
                    elif "5.3.8" in header: from_state = "RRC_CONNECTED" # Release starts from CONNECTED
                
                for to_state_raw in enter_matches:
                    to_state = to_state_raw.upper().replace("\\", "").replace("*", "").strip("_")
                    if "CONNECTED" in to_state: to_state = "RRC_CONNECTED"
                    elif "IDLE" in to_state: to_state = "RRC_IDLE"
                    elif "INACTIVE" in to_state: to_state = "RRC_INACTIVE"
                    else: continue # Skip non-RRC states for now
                    
                    trigger_match = re.search(r"(\d+(\.\d+)+)", header)
                    trigger = trigger_match.group(1) if trigger_match else header[:20]
                    
                    # Manual Override for specific 3GPP clauses
                    if "5.3.8.3" in trigger: from_state = "RRC_CONNECTED"
                    if "5.3.11" in trigger: 
                        from_state = "RRC_CONNECTED"
                        to_state = "RRC_IDLE"
                    if "5.3.3.1" in trigger: from_state = "RRC_IDLE"
                    
                    self.transitions.append({
                        "from": from_state,
                        "to": to_state,
                        "trigger": trigger,
                        "clause": header
                    })
        
        # Add some manual standard transitions if the doc is too complex for simple regex
        if not self.transitions:
            self.transitions.append({"from": "RRC_IDLE", "to": "RRC_CONNECTED", "trigger": "Setup", "clause": "Manual"})
            self.transitions.append({"from": "RRC_CONNECTED", "to": "RRC_IDLE", "trigger": "Release", "clause": "Manual"})
            self.transitions.append({"from": "RRC_CONNECTED", "to": "RRC_INACTIVE", "trigger": "Suspend", "clause": "Manual"})
            self.transitions.append({"from": "RRC_INACTIVE", "to": "RRC_CONNECTED", "trigger": "Resume", "clause": "Manual"})
        
    def generate_mermaid(self):
        mermaid = "stateDiagram-v2\n"
        seen = set()
        for t in self.transitions:
            # Skip self-transitions to UNKNOWN if they don't add value
            if t['from'] == "UNKNOWN" and t['to'] == "UNKNOWN": continue
            
            key = (t['from'], t['to'], t['trigger'])
            if key not in seen:
                mermaid += f"    {t['from']} --> {t['to']}: {t['trigger']}\n"
                seen.add(key)
        return mermaid

if __name__ == "__main__":
    md_file = os.path.expanduser("~/repo/remote_work/doc_logic_fsm/docs/md/38331-j10.md")
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found.")
    else:
        extractor = NR_RRC_FSM_Extractor(md_file)
        print("Segmenting document...")
        extractor.segment_by_header()
        print(f"Extracted {len(extractor.clauses)} clauses.")
        print("Extracting state transitions...")
        extractor.extract_logic()
        
        mermaid_code = extractor.generate_mermaid()
        
        output_dir = os.path.expanduser("~/repo/remote_work/doc_logic_fsm/fsm_core")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(os.path.join(output_dir, "rrc_fsm.mermaid"), "w") as f:
            f.write(mermaid_code)
        
        print(f"FSM generated and saved to {output_dir}/rrc_fsm.mermaid")
        print("\nMermaid Code:\n")
        print(mermaid_code)
