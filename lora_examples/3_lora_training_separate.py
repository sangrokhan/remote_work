import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleModel
from dataset import get_dataloader
from lora import apply_lora, get_lora_params

def train_lora_separate(epochs=5, output_file="lora_adapter.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleModel().to(device)
    try:
        model.load_state_dict(torch.load("base_model.pth"))
    except FileNotFoundError:
        pass

    apply_lora(model)
    model.to(device)
    
    lora_params = get_lora_params(model)
    optimizer = optim.Adam(lora_params, lr=0.01)
    criterion = nn.MSELoss()
    dataloader = get_dataloader()
    
    print(f"Start LoRA Training (Separate Save to {output_file})...")
    for epoch in range(epochs):
        model.train()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            
    # Save ONLY LoRA params
    # We grab the state_dict of the FULL model, but filter for keys containing 'lora_'
    full_state_dict = model.state_dict()
    lora_state_dict = {k: v for k, v in full_state_dict.items() if 'lora_' in k}
    
    torch.save(lora_state_dict, output_file)
    print(f"LoRA adapter saved to {output_file}")

if __name__ == "__main__":
    train_lora_separate(epochs=5)
