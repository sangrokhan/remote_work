import torch
from model import SimpleModel
from lora import apply_lora
import copy

def inference_merge_multiple():
    # Setup: We need two adapters. Let's assume we have lora_adapter_A.pth and lora_adapter_B.pth
    # For this example, we'll just simulate it by saving the same adapter twice if they don't exist.
    import os
    if not os.path.exists("lora_adapter_A.pth"):
        if os.path.exists("lora_adapter.pth"):
            import shutil
            shutil.copy("lora_adapter.pth", "lora_adapter_A.pth")
            shutil.copy("lora_adapter.pth", "lora_adapter_B.pth")
            print("Created dummy adapters A and B from lora_adapter.pth")
        else:
            print("Please run 3_lora_training_separate.py first.")
            return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    model.load_state_dict(torch.load("base_model.pth"))
    
    apply_lora(model)
    model.to(device)
    
    # Load Adapter A
    state_A = torch.load("lora_adapter_A.pth")
    # Load Adapter B
    state_B = torch.load("lora_adapter_B.pth")
    
    print("Merging Adapter A and B with 0.5 weight each...")
    
    # We will manually load them into the model.
    # Approach: Load A, merge. Load B, merge?
    # No, if we merge A, the base weights change. if we then load B (which assumes original base), and merge, it might be additive.
    # LoRA is additive: W_new = W + dW_A + dW_B.
    # So yes, we can load A, merge, then load B (into the LoRA layers which are now reset? No)
    # The `merge()` method adds (B@A)*scale to original weights and sets merged=True.
    
    # Correct Multi-LoRA Merge Pattern:
    # 1. Load A into LoRA layers.
    model.load_state_dict(state_A, strict=False)
    
    # 2. Merge A.
    for m in model.modules():
        if hasattr(m, 'merge'):
             # We might want to scale A by 0.5?
             # m.scaling *= 0.5  <-- simple hack
             m.merge()
             
    # 3. Reset LoRA layers to be ready for next adapter?
    # merge() keeps merged=True. We need to unmerge? No, we want to ADD B.
    # But we can't just load B into lora_A/lora_B parameters because `merge` checks `self.merged`.
    # We need to flip `merged` back to False BUT `original_layer` now holds W+A.
    # So if we load B and merge, we get (W+A) + B. This is correct for additive LoRA.
    
    for m in model.modules():
        if hasattr(m, 'merged'):
            m.merged = False
            # Reset params to 0 so we don't double count if load fails? not needed if we load state_dict immediately.
            
    # 4. Load B
    model.load_state_dict(state_B, strict=False)
    
    # 5. Merge B
    for m in model.modules():
        if hasattr(m, 'merge'):
             m.merge()
             
    print("Merged multiple adapters.")
    model.eval()
    print(model(torch.randn(1, 10).to(device)))

if __name__ == "__main__":
    inference_merge_multiple()
