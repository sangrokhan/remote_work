import os
import sys

def generate_html_viewer(mermaid_path, html_path):
    """
    Mermaid.js를 사용하여 Mermaid 다이어그램을 시각화하는 HTML 파일을 생성합니다.
    """
    try:
        with open(mermaid_path, 'r', encoding='utf-8') as f:
            mermaid_code = f.read()
            
        html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>RRC State Machine Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{ startOnLoad: true }});
    </script>
    <style>
        body {{ font-family: sans-serif; display: flex; flex-direction: column; align-items: center; padding: 20px; }}
        .mermaid {{ width: 100%; max-width: 1000px; }}
        h1 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>3GPP NR RRC State Machine</h1>
    <div class="mermaid">
{mermaid_code}
    </div>
</body>
</html>
"""
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        return True
    except Exception as e:
        print(f"Error generating visualizer: {e}")
        return False

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    mermaid_file = os.path.join(base_dir, "fsm_core", "rrc_fsm.mermaid")
    output_file = os.path.join(base_dir, "validation", "fsm_viewer.html")
    
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
        
    if generate_html_viewer(mermaid_file, output_file):
        print(f"Visualization HTML generated at {output_file}")
