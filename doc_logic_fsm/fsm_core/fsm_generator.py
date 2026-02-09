import re
import json
import os
import spacy
from collections import Counter

class GenericProtocolFSMExtractor:
    """
    표준 문서에서 상태(Node)와 전이(Edge)를 스스로 발견하여 FSM을 생성하는 일반화된 추출 엔진입니다.
    """
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
        """
        Markdown 헤더 구조를 분석하여 문서를 절(Clause) 단위로 분할합니다.
        """
        with open(self.md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 문서 구조 파싱 (일반적인 표준 문서의 계층적 헤더 패턴)
        header_pattern = r"(^#{1,5}\s+\d+[\d\.]*\s+.*)"
        parts = re.split(header_pattern, content, flags=re.MULTILINE)
        
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            body = parts[i+1] if i+1 < len(parts) else ""
            self.clauses[header] = body

    def discover_states(self):
        """
        문서 전체 텍스트를 분석하여 'state'라는 단어와 결합된 고유 명사들을 찾아내어 
        이를 FSM의 노드(State)로 자동 등록합니다.
        """
        all_text = " ".join(list(self.clauses.values()))
        
        # 1. 'X state' 또는 'state X' 패턴 추출 (X는 대문자로 시작하는 고유 명사 형태)
        # 예: RRC_CONNECTED state, state IDLE 등
        candidates = []
        
        # 패턴 1: 대문자 위주의 명사가 'state' 앞에 오는 경우
        candidates += re.findall(r"([A-Z][A-Z0-9_]{2,})\s+state", all_text)
        
        # 패턴 2: 'state' 뒤에 대문자 위주의 명사가 오는 경우
        candidates += re.findall(r"state\s+([A-Z][A-Z0-9_]{2,})", all_text)
        
        # 2. 통계적 필터링: 빈도가 높거나 언더바(_)를 포함한 기술적 용어를 상태로 간주
        counts = Counter(candidates)
        self.discovered_states = {s for s, count in counts.items() if count > 2 or "_" in s}
        
        # 발견된 상태 이름 정규화 (Artifact 제거)
        self.discovered_states = {s.strip("_").replace("\\", "") for s in self.discovered_states}
        
        print(f"[*] Discovered {len(self.discovered_states)} potential states: {self.discovered_states}")

    def extract_logic(self):
        """
        발견된 상태(States)를 바탕으로 각 절에서 발생하는 전이(Transitions)를 추출합니다.
        """
        self.discover_states()
        
        # 전이 유발 키워드 (언어적 패턴)
        transition_verbs = ["enter", "transition to", "going to", "move to", "return to"]
        
        for header, body in self.clauses.items():
            # 텍스트 클리닝 (Markdown artifacts 제거)
            body_clean = body.replace("\\_", "_")
            header_clean = header.replace("\\_", "_")
            
            # 1. 현재 맥락 상태(Context State) 파악
            # 해당 절이 어떤 상태에서 시작되는지 문맥을 통해 판단
            from_state = "START_OR_UNKNOWN"
            for s in self.discovered_states:
                if f"UE in {s}" in body_clean or f"in {s} state" in body_clean:
                    from_state = s
                    break
            
            # 2. 문장 단위 분석을 통한 전이 발견
            # Spacy를 활용하여 문장을 나누고 전이 로직 검색
            doc = self.nlp(body_clean[:3000]) # 성능을 위해 절의 초반부 집중 분석
            for sent in doc.sents:
                text = sent.text
                
                # 전이 동사가 포함된 문장인지 확인
                if any(verb in text.lower() for verb in transition_verbs):
                    for to_state in self.discovered_states:
                        if to_state in text and to_state != from_state:
                            # 3. 트리거(Trigger) 식별: 현재 절의 번호나 제목을 트리거로 사용
                            trigger_match = re.search(r"(\d+(\.\d+)+)", header_clean)
                            trigger = trigger_match.group(1) if trigger_match else "Trigger"
                            
                            self.transitions.append({
                                "from": from_state,
                                "to": to_state,
                                "trigger": trigger,
                                "evidence": text.strip()[:50] + "..."
                            })

    def generate_mermaid(self):
        """
        추출된 전이 데이터를 Mermaid 형식의 다이어그램 코드로 변환합니다.
        """
        mermaid = "stateDiagram-v2\n"
        # 중복 전이 제거 및 가독성을 위한 정렬
        seen = set()
        for t in self.transitions:
            key = (t['from'], t['to'], t['trigger'])
            if key not in seen:
                mermaid += f"    {t['from']} --> {t['to']}: {t['trigger']}\n"
                seen.add(key)
        
        if len(seen) == 0:
            mermaid += "    Note right of START_OR_UNKNOWN: No transitions discovered automatically.\n"
            
        return mermaid

if __name__ == "__main__":
    md_file = os.path.expanduser("~/repo/remote_work/doc_logic_fsm/docs/md/38331-j10.md")
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found.")
    else:
        extractor = GenericProtocolFSMExtractor(md_file)
        print("[1/3] Segmenting document clauses...")
        extractor.segment_by_header()
        
        print("[2/3] Analyzing and discovering FSM structure...")
        extractor.extract_logic()
        
        print("[3/3] Generating Mermaid visualization...")
        mermaid_code = extractor.generate_mermaid()
        
        output_dir = os.path.expanduser("~/repo/remote_work/doc_logic_fsm/fsm_core")
        with open(os.path.join(output_dir, "rrc_fsm.mermaid"), "w") as f:
            f.write(mermaid_code)
        
        print(f"\n[Success] FSM generated at: {output_dir}/rrc_fsm.mermaid")
        print("-" * 30)
        print(mermaid_code)
        print("-" * 30)
