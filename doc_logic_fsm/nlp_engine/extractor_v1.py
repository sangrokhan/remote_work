import re
import json

class SimpleLogicExtractor:
    def __init__(self):
        # Initial patterns for 5G NR RRC
        self.trigger_patterns = [
            r"upon reception of (the\s+)?(?P<msg>[\w-]+)",
            r"receipt of (the\s+)?(?P<msg>[\w-]+)"
        ]
        self.action_patterns = [
            r"stop timer (?P<timer>T\d+)",
            r"enter (?P<state>[\w_]+) state",
            r"submit (the\s+)?(?P<msg>[\w-]+) message"
        ]

    def extract(self, text):
        logic = {"triggers": [], "actions": [], "state_transitions": []}
        
        # Extract Triggers
        for pattern in self.trigger_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                logic["triggers"].append(match.group("msg"))

        # Extract Actions & State Transitions
        for pattern in self.action_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if "timer" in match.groupdict():
                    logic["actions"].append(f"STOP_TIMER_{match.group('timer')}")
                if "state" in match.groupdict():
                    logic["state_transitions"].append(match.group("state"))
                if "msg" in match.groupdict():
                    logic["actions"].append(f"SUBMIT_{match.group('msg')}")

        return logic

if __name__ == "__main__":
    extractor = SimpleLogicExtractor()
    sample_texts = [
        "The UE shall perform the following actions upon reception of the RRCSetup:",
        "1> stop timer T300, if running;",
        "2> enter RRC_CONNECTED state;",
        "2> submit the RRCSetupComplete message to lower layers for transmission;"
    ]
    
    results = []
    for t in sample_texts:
        res = extractor.extract(t)
        if any(res.values()):
            results.append({"text": t, "logic": res})
            
    print(json.dumps(results, indent=2))
