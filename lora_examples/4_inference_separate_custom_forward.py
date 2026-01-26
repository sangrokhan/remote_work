import torch
from model import SimpleModel
from lora import apply_lora

def inference_separate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Base Model
    model = SimpleModel().to(device)
    try:
        model.load_state_dict(torch.load("base_model.pth"))
        print("Loaded base model.")
    except:
        print("Warning: base_model.pth not found. Run 1_basic_training.py first.")
        return

    # 2. Apply LoRA (structure only)
    apply_lora(model)
    model.to(device)
    
    # 3. Load LoRA weights
    try:
        lora_state_dict = torch.load("lora_adapter.pth")
        
        # We need to be careful with keys. 
        # If we saved using the filtering method in step 3, keys are like 'layer1.lora_A', etc.
        # This matches the model structure after apply_lora().
        missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)
        print(f"Loaded LoRA weights. Missing (should be base params): {len(missing)}, Unexpected: {len(unexpected)}")
    except:
        print("Warning: lora_adapter.pth not found. Run 3_lora_training_separate.py first.")
        return

    # 4. Inference
    model.eval()
    dummy_input = torch.randn(1, 10).to(device)
    
    with torch.no_grad():
        # The forward pass in LoRALinear automatically does: base(x) + lora(x) * scale
        output = model(dummy_input)
    
    print("Inference Output:", output)

if __name__ == "__main__":
    inference_separate()
