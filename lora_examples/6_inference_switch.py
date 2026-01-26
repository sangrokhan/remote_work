import torch
from model import SimpleModel
from lora import apply_lora
import copy

class MultiLoRAManager:
    def __init__(self, model):
        self.model = model
        self.adapters = {} # name -> state_dict
        
    def add_adapter(self, name, state_dict):
        self.adapters[name] = state_dict
        
    def activate_adapter(self, name):
        if name not in self.adapters:
            print(f"Adapter {name} not found.")
            return
        
        print(f"Switching to adapter: {name}")
        # Load the state dict into the model
        # strict=False because state_dict only has LoRA params
        self.model.load_state_dict(self.adapters[name], strict=False)

def inference_switch():
    # Setup dummy adapters
    import os
    if not os.path.exists("lora_adapter.pth"):
        print("Run 3_lora_training_separate.py first.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    model.load_state_dict(torch.load("base_model.pth"))
    
    apply_lora(model)
    model.eval()
    
    # Create Manager
    manager = MultiLoRAManager(model)
    
    # Load same adapter as 'A' and 'B' for demo
    adapter_weights = torch.load("lora_adapter.pth")
    manager.add_adapter("task_A", adapter_weights)
    
    # Create a fake "Task B" by modifying weights slightly just to see difference
    adapter_weights_b = copy.deepcopy(adapter_weights)
    for k in adapter_weights_b:
        adapter_weights_b[k] = adapter_weights_b[k] * -1 # Invert weights
    manager.add_adapter("task_B", adapter_weights_b)
    
    # Input
    x = torch.randn(1, 10).to(device)
    
    # Run Task A
    manager.activate_adapter("task_A")
    out_a = model(x)
    print(f"Output A: {out_a.item():.4f}")
    
    # Run Task B
    manager.activate_adapter("task_B")
    out_b = model(x)
    print(f"Output B: {out_b.item():.4f}")

if __name__ == "__main__":
    inference_switch()
