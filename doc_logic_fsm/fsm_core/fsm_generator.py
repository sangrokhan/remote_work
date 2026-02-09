import re
import json
import os
import spacy
from collections import Counter

class AutonomousFSMDiscoverer:
    """
    표준 문서의 언어적 구조와 엔티티 간의 관계를 분석하여 
    상태(Node)와 전이(Edge)를 스스로 발견하는 엔진입니다.
    특정 키워드(state) 없이도 '상태'의 논리적 특성을 학습합니다.
    """
    def __init__(self, md_path):
        self.md_path = md_path
        self.clauses = {}
        self.transitions = []
        self.nodes = set()
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    def load_and_segment(self):
        with open(self.md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 3GPP Artifact cleanup
        content = content.replace("\\_", "_").replace("*", "")
        
        header_pattern = r"(^#{1,5}\s+\d+[\d\.]*\s+.*)"
        parts = re.split(header_pattern, content, flags=re.MULTILINE)
        for i in range(1, len(parts), 2):
            self.clauses[parts[i].strip()] = parts[i+1] if i+1 < len(parts) else ""

    def discover_nodes(self):
        """
        '상태'의 본질적 역할(전이 동작의 목적지)을 분석하여 노드를 식별합니다.
        """
        print("[*] Discovering logical nodes through behavior analysis...")
        verbs = ["enter", "transition", "move", "remain", "resume", "suspend", "release"]
        candidates = Counter()
        
        # 제외 목록: 프로토콜 공통 용어 및 문법적 노이즈
        blacklist = {
            "3GPP", "TS", "NR", "UE", "EUTRA", "RAT", "AMF", "RAN", "SIB", "MAC", "PDCP", "RLC", "SRB", "DRB", "LTE", 
            "CELL", "NETWORK", "PCELL", "THE", "ALL", "ANY", "EACH", "THIS", "THAT", "SHALL", "WILL", "PROCEDURE", 
            "CASE", "INFORMATION", "VALUE", "VALUES", "MESSAGE", "MESSAGES", "CONTENTS", "CONFIGURATION", 
            "CONDITION", "CONDITIONS", "TIMER", "TIMERS", "START", "STOP", "END", "EXIT", "ENTRY", "ENTRIES", 
            "FOLLOWING", "BELOW", "ABOVE", "DUE", "WITH", "WITHOUT", "FROM", "INTO", "VIA", "OTHER", "ANOTHER"
        }

        for body in self.clauses.values():
            body_upper = body.upper()
            for verb in verbs:
                # 전이 동사와 결합된 대문자 단어군 추출
                matches = re.findall(rf"{verb}s?\s+(?:to\s+)?(?:the\s+)?([A-Z0-9_]{{3,}})", body, re.IGNORECASE)
                for m in matches:
                    name = m.upper().strip("_")
                    if name not in blacklist and len(name) > 2:
                        candidates[name] += 1
        
        # RRC_CONNECTED, RRC_IDLE 등 3GPP 표준 상태 패턴 가중치
        for body in self.clauses.values():
            rrc_states = re.findall(r"RRC_[A-Z_]+", body)
            for s in rrc_states:
                candidates[s.upper().replace("RRC_", "")] += 2

        # 상위 빈도 엔티티 확정
        self.nodes = {name for name, count in candidates.most_common(15)}
        print(f"[*] Identified potential states: {self.nodes}")

    def extract_transitions(self):
        """식별된 노드 간의 실제 전이 명령을 본문에서 추출합니다."""
        print("[*] Mapping transitions between discovered nodes...")
        
        for header, body in self.clauses.items():
            body_upper = body.upper()
            
            # From-Node 추론: 현재 절의 맥락 파악
            from_node = "START_OR_GLOBAL"
            for node in self.nodes:
                if f"UE IN {node}" in body_upper or f"WHILE IN {node}" in body_upper:
                    from_node = node
                    break
            
            # To-Node 추론: 전이 키워드 결합 분석
            for target in self.nodes:
                if target == from_node: continue
                
                # 'enter [State]' 또는 'transition to [State]'
                if re.search(rf"(enter|transition|move|resume|suspend)\s+(?:to\s+)?(?:the\s+)?(?:RRC_)?{target}", body, re.IGNORECASE):
                    trigger = re.search(r"(\d+(\.\d+)+)", header)
                    trigger_text = trigger.group(1) if trigger else header[:15]
                    
                    self.transitions.append({
                        "from": from_node,
                        "to": target,
                        "trigger": trigger_text
                    })

    def generate_mermaid(self):
        mermaid = "stateDiagram-v2\n"
        seen = set()
        for t in self.transitions:
            edge = f"    {t['from']} --> {t['to']}: {t['trigger']}\n"
            if edge not in seen:
                mermaid += edge
                seen.add(edge)
        return mermaid

if __name__ == "__main__":
    md_file = os.path.expanduser("~/repo/remote_work/doc_logic_fsm/docs/md/38331-j10.md")
    if not os.path.exists(md_file):
        print(f"Error: Spec file not found.")
    else:
        engine = AutonomousFSMDiscoverer(md_file)
        engine.load_and_segment()
        engine.discover_nodes()
        engine.extract_transitions()
        
        code = engine.generate_mermaid()
        output_path = os.path.expanduser("~/repo/remote_work/doc_logic_fsm/fsm_core/rrc_fsm.mermaid")
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(code)
        
        print("\nDiscovered Logic Flow:")
        print("-" * 30)
        print(code)
        print("-" * 30)
