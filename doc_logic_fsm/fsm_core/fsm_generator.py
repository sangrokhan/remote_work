import re
import json
import os
import spacy
from collections import Counter

class GenericProtocolFSMExtractor:
    def __init__(self, md_path):
        self.md_path = md_path
        self.clauses = {}
        self.transitions = []
        self.discovered_states = set()
        
        # Load NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spacy model...")
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    def segment_by_header(self):
        with open(self.md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # General pattern for standard document clauses (e.g., # 5.3.3)
        header_pattern = r"(^#{1,5}\s+\d+[\d\.]*\s+.*)"
        parts = re.split(header_pattern, content, flags=re.MULTILINE)
        
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            body = parts[i+1] if i+1 < len(parts) else ""
            self.clauses[header] = body

    def discover_states(self):
        """
        문서 전체에서 'state' 단어와 함께 등장하는 고유 명사를 찾아 상태 후보로 등록합니다.
        """
        # Look at more text to find all potential states
        all_text = " ".join(list(self.clauses.values())) 
        
        # Pattern: [Upper_Case_Name] state (e.g., RRC_CONNECTED state)
        # 3GPP uses RRC_IDLE, RRC_CONNECTED etc.
        state_candidates = re.findall(r"([A-Z][A-Z0-9_]{3,})\s+state", all_text)
        
        # Pattern: state [Upper_Case_Name]
        state_candidates += re.findall(r"state\s+([A-Z][A-Z0-9_]{3,})", all_text)
        
        counts = Counter(state_candidates)
        # Register candidates that appear multiple times or look like standard states
        self.discovered_states = {s for s, count in counts.items() if count > 5}
        print(f"Discovered potential states: {self.discovered_states}")

    def extract_logic(self):
        self.discover_states()
        
        for header, body in self.clauses.items():
            body_clean = body.replace("\\_", "_")
            header_clean = header.replace("\\_", "_")
            
            # Find the context state for this clause (from where we start)
            from_state = "UNKNOWN"
            # Strategy: look for "UE in [STATE]" or states mentioned in the header
            for s in self.discovered_states:
                if s in header_clean or f"UE in {s}" in body_clean[:1000]:
                    from_state = s
                    break
            
            # Look for transition indicators: "enter [STATE]", "going to [STATE]"
            for s in self.discovered_states:
                # Dynamic regex for 'enter [STATE]' or 'going to [STATE]'
                patterns = [
                    rf"enter\s+(?:the\s+)?\*?(?:RRC_)?{s}\*?",
                    rf"going\s+to\s+\*?(?:RRC_)?{s}\*?"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, body_clean, re.IGNORECASE)
                    if match:
                        # Attempt to refine from_state if still UNKNOWN by looking at previous mentions
                        if from_state == "UNKNOWN":
                            # Look for any state mentioned before the match
                            pre_match_text = body_clean[:match.start()]
                            for prev_s in self.discovered_states:
                                if prev_s in pre_match_text and prev_s != s:
                                    from_state = prev_s
                        
                        if from_state == s: continue # Avoid self-loops for now unless explicit
                        
                        trigger_match = re.search(r"(\d+(\.\d+)+)", header_clean)
                        trigger = trigger_match.group(1) if trigger_match else header_clean[:20]
                        
                        self.transitions.append({
                            "from": from_state,
                            "to": s,
                            "trigger": trigger
                        })
                        break

    def generate_mermaid(self):
        mermaid = "stateDiagram-v2\n"
        seen = set()
        for t in self.transitions:
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
        extractor = GenericProtocolFSMExtractor(md_file)
        print("Segmenting document...")
        extractor.segment_by_header()
        print(f"Extracted {len(extractor.clauses)} clauses.")
        print("Extracting state transitions (Generalized)...")
        extractor.extract_logic()
        
        mermaid_code = extractor.generate_mermaid()
        
        output_dir = os.path.expanduser("~/repo/remote_work/doc_logic_fsm/fsm_core")
        with open(os.path.join(output_dir, "rrc_fsm.mermaid"), "w") as f:
            f.write(mermaid_code)
        
        print(f"FSM generated and saved to {output_dir}/rrc_fsm.mermaid")
        print("\nMermaid Code:\n")
        print(mermaid_code)
