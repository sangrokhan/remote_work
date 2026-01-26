import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleModel
from dataset import get_dataloader
from lora import apply_lora, get_lora_params

def train_lora_merge(epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Base Model
    model = SimpleModel().to(device)
    try:
        model.load_state_dict(torch.load("base_model.pth"))
        print("Loaded base model weights.")
    except FileNotFoundError:
        print("Base model not found, initializing random.")

    # 2. Apply LoRA
    # We only want to apply LoRA to linear layers.
    apply_lora(model)
    model.to(device)
    
    # 3. Freeze non-LoRA parameters (already done in LoRALinear init but good to double check or be explicit)
    # The helper `get_lora_params` gives us just the trainable bits.
    lora_params = get_lora_params(model)
    
    # Verify parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in lora_params)
    print(f"Total Params: {total_params}, Trainable (LoRA) Params: {trainable_params}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lora_params, lr=0.01) # Higher LR for LoRA usually
    
    dataloader = get_dataloader(size=1000)
    
    print("Start LoRA Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
        
    # 4. Merge Weights
    # Iterate through modules and call merge() if it's a LoRALinear
    print("Merging LoRA weights...")
    for module in model.modules():
        if hasattr(module, 'merge'):
            module.merge()
            
    # 5. Save Merged Model
    # Now the state_dict will contain the merged weights with the same keys as original model
    # Note: LoRALinear replaces the original Linear, so the keys might have changed if we are not careful.
    # But wait, our LoRALinear WRAPS the original layer. 
    # The state_dict will have keys like 'layer1.original_layer.weight', 'layer1.lora_A', etc.
    # To save it compatible with the original SimpleModel, we need to extract the weights from the wrapped original_layer.
    
    merged_state_dict = {}
    for name, module in model.named_children():
        if hasattr(module, 'original_layer'): # It is a LoRA layer
            # merged weights are in module.original_layer.weight
            merged_state_dict[f"{name}.weight"] = module.original_layer.weight.data
            merged_state_dict[f"{name}.bias"] = module.original_layer.bias.data
        else:
            # Just copy standard layers (like ReLU or others if any)
            # In our simple model, everything is a linear layer or relu (which has no state).
            pass
            
    torch.save(merged_state_dict, "lora_merged_model.pth")
    print("Merged model saved to lora_merged_model.pth")

if __name__ == "__main__":
    train_lora_merge()
