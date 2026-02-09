import json
import os
import subprocess

def test_pipeline():
    # 1. Preprocessing (Segmentation)
    print("Testing Preprocessing...")
    res_seg = subprocess.run(["python3", "preprocessing/segmenter.py"], capture_output=True, text=True)
    assert res_seg.returncode == 0
    print("OK")

    # 2. NLP Extraction (v3)
    print("Testing NLP Extraction...")
    res_ext = subprocess.run(["python3", "nlp_engine/extractor_v3.py"], capture_output=True, text=True)
    assert res_ext.returncode == 0
    assert os.path.exists("fsm_core/multi_layer_logic.json")
    print("OK")

    # 3. Visualization
    print("Testing Visualization...")
    res_vis = subprocess.run(["python3", "validation/interactor.py"], capture_output=True, text=True)
    assert res_vis.returncode == 0
    assert os.path.exists("validation/interaction_diagram.mmd")
    print("OK")

    print("\nAll pipeline tests PASSED!")

if __name__ == "__main__":
    test_pipeline()
