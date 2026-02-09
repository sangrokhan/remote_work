import re
import json
import os

class MultiLayerLogicExtractor:
    def __init__(self):
        # Patterns for multi-layer interaction
        self.trigger_patterns = [
            (r"triggers the (?P<proc>[\w\s]+) procedure", "INTER_LAYER_TRIGGER"),
            (r"upon reception of (the\s+)?(?P<msg>[\w-]+)", "MSG_RECEIVE"),
            (r"If the (?P<msg>[\w\s-]+) is successfully received", "MSG_SUCCESS")
        ]
        self.action_patterns = [
            (r"instruct (the\s+)?(?P<entity>[\w\s]+) to (?P<act>.+)", "CROSS_LAYER_ACT"),
            (r"stop timer (?P<timer>T\d+)", "TIMER_STOP"),
            (r"set the (?P<var>[\w_]+) to (?P<val>.+)", "VAR_SET"),
            (r"indicate the (?P<evt>.+?) to (?P<target>upper layers|lower layers)", "INTER_LAYER_IND")
        ]

    def extract(self, line):
        results = []
        line = line.strip()
        
        for pattern, tag in self.trigger_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                results.append({"type": tag, "data": match.groupdict()})
        
        for pattern, tag in self.action_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                results.append({"type": tag, "data": match.groupdict()})
        
        return results

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extractor = MultiLayerLogicExtractor()
    files = [
        os.path.join(base_dir, "docs/TS38331_snippet.txt"),
        os.path.join(base_dir, "docs/TS38321_snippet.txt")
    ]
    
    all_logic = {}
    for f_path in files:
        with open(f_path, 'r') as f:
            lines = f.readlines()
        
        doc_name = os.path.basename(f_path)
        all_logic[doc_name] = []
        for l in lines:
            extracted = extractor.extract(l)
            if extracted:
                all_logic[doc_name].append({"text": l.strip(), "logic": extracted})
                
    output_path = os.path.join(base_dir, "fsm_core/multi_layer_logic.json")
    with open(output_path, 'w') as f:
        json.dump(all_logic, f, indent=2)
    print(f"Multi-layer logic extracted to {output_path}")
