import os
import subprocess

def test_pipeline():
    # Resolve project root relative to this file
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    os.chdir(base_dir)

    print(f"[*] Project root: {os.getcwd()}")

    # 1. FSM Generation
    print("[*] Testing FSM Structure Discovery...")
    res_gen = subprocess.run(["python3", "fsm_core/fsm_generator.py"], capture_output=True, text=True)
    if res_gen.returncode != 0:
        print(f"Error: {res_gen.stderr}")
    assert res_gen.returncode == 0
    assert os.path.exists("fsm_core/rrc_fsm.json")
    print("OK")

    # 2. Visualization
    print("[*] Testing Visualizer...")
    res_vis = subprocess.run(["python3", "validation/visualizer.py"], capture_output=True, text=True)
    if res_vis.returncode != 0:
        print(f"Error: {res_vis.stderr}")
    assert res_vis.returncode == 0
    assert os.path.exists("validation/fsm_viewer.html")
    print("OK")

    print("\nAll new pipeline tests PASSED!")

if __name__ == "__main__":
    test_pipeline()
