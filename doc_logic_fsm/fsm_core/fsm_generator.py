import re
import json
import os
import spacy
from collections import Counter

class AutonomousProtocolDiscovery:
    """
    사전에 정의된 키워드나 상태 리스트 없이, 
    표준 문서의 언어적 구조와 엔티티 간의 관계를 분석하여 
    상태(Node)와 전이(Edge)를 스스로 '발견'하는 엔진입니다.
    """
    def __init__(self, md_path):
        self.md_path = md_path
        self.clauses = {}
        self.transitions = []
        self.entities = Counter()
        self.nodes = set()
        
        # 언어 분석을 위한 NLP 모델 로드
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    def load_and_segment(self):
        """문서를 구조적 계층에 따라 분할합니다."""
        with open(self.md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 표준 문서의 섹션 패턴 (# 5.3.3 등)
        header_pattern = r"(^#{1,5}\s+\d+[\d\.]*\s+.*)"
        parts = re.split(header_pattern, content, flags=re.MULTILINE)
        
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            body = parts[i+1] if i+1 < len(parts) else ""
            self.clauses[header] = body

    def discover_nodes(self):
        """
        '상태'로 의심되는 엔티티들을 스스로 발견합니다.
        기준: 'enter', 'transition to', 'remain in' 등의 동사와 함께 쓰이는 고유 명사군
        """
        print("[*] Phase 1: Discovering logical nodes...")
        candidate_patterns = [
            r"enter(?:s|ing)?\s+(?:the\s+)?\*?([A-Z][A-Z0-9_]{3,})\*?",
            r"transition(?:s)?\s+to\s+(?:the\s+)?\*?([A-Z][A-Z0-9_]{3,})\*?",
            r"remain(?:s)?\s+in\s+(?:the\s+)?\*?([A-Z][A-Z0-9_]{3,})\*?",
            r"leaving\s+(?:the\s+)?\*?([A-Z][A-Z0-9_]{3,})\*?",
            r"in\s+\*?([A-Z][A-Z0-9_]{3,})\*?\s+state"
        ]
        
        for body in self.clauses.values():
            body_clean = body.replace("\\_", "_")
            for pattern in candidate_patterns:
                matches = re.findall(pattern, body_clean)
                for m in matches:
                    # RRC_CONNECTED -> CONNECTED (정규화)
                    clean_m = m.upper().strip("_").replace("RRC_", "")
                    if len(clean_m) > 2: # 너무 짧은 엔티티 제외
                        self.entities[clean_m] += 1

        # 일정 빈도 이상 등장하거나 특정 패턴을 가진 엔티티를 Node로 확정
        self.nodes = {name for name, count in self.entities.items() if count >= 1}
        print(f"[*] Discovered Nodes: {self.nodes}")

    def analyze_transitions(self):
        """
        발견된 Node들 간의 관계(Edge)를 의미론적으로 추출합니다.
        """
        print("[*] Phase 2: Analyzing semantic transitions...")
        
        for header, body in self.clauses.items():
            body_clean = body.replace("\\_", "_")
            header_clean = header.replace("\\_", "_")
            
            # 1. Clause의 시작 맥락(From-Node) 발견
            context_node = "UNKNOWN"
            # 본문에서 'UE in [NODE]' 패턴 검색
            for node in self.nodes:
                if f"UE IN {node}" in body_clean.upper() or f"IN {node} STATE" in body_clean.upper():
                    context_node = node
                    break
            
            # 2. 전이 액션(To-Node) 발견
            # Spacy를 사용하여 문장 단위로 정밀 분석
            doc = self.nlp(body_clean[:2000]) 
            for sent in doc.sents:
                sent_text = sent.text.upper()
                
                # 'ENTER', 'GO TO', 'TRANSITION' 등의 의미를 가진 문장인지 확인
                if any(verb in sent_text for verb in ["ENTER", "TRANSITION", "MOVE", "GO TO", "RESUME", "ESTABLISH"]):
                    for target_node in self.nodes:
                        if target_node in sent_text and target_node != context_node:
                            # 트리거 식별 (절 번호 등)
                            trigger_match = re.search(r"(\d+(\.\d+)+)", header_clean)
                            trigger = trigger_match.group(1) if trigger_match else "Procedure"
                            
                            self.transitions.append({
                                "from": context_node,
                                "to": target_node,
                                "trigger": trigger
                            })

    def export_mermaid(self, output_path):
        """결과를 Mermaid 코드로 변환하여 저장합니다."""
        mermaid = "stateDiagram-v2\n"
        seen = set()
        for t in self.transitions:
            # 중복 제거 및 가독성 필터링
            if t['from'] == "UNKNOWN" and len(self.transitions) > 10: # 데이터가 많으면 UNKNOWN 발산은 제외
                pass 
            
            key = (t['from'], t['to'], t['trigger'])
            if key not in seen:
                mermaid += f"    {t['from']} --> {t['to']}: {t['trigger']}\n"
                seen.add(key)
        
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(mermaid)
        return mermaid

if __name__ == "__main__":
    md_file = os.path.expanduser("~/repo/remote_work/doc_logic_fsm/docs/md/38331-j10.md")
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found.")
    else:
        engine = AutonomousProtocolDiscovery(md_file)
        engine.load_and_segment()
        engine.discover_nodes()
        engine.analyze_transitions()
        
        mermaid_path = os.path.expanduser("~/repo/remote_work/doc_logic_fsm/fsm_core/rrc_fsm.mermaid")
        code = engine.export_mermaid(mermaid_path)
        
        print("\n" + "="*50)
        print("AUTONOMOUS FSM DISCOVERY COMPLETE")
        print("="*50)
        print(code)
        print("="*50)
