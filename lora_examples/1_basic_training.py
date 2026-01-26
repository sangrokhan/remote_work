import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleModel
from dataset import get_dataloader

def train(epochs=5):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init model & data
    model = SimpleModel().to(device)
    dataloader = get_dataloader(size=1000)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Start Basic Training...")
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
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
    # Save base model
    torch.save(model.state_dict(), "base_model.pth")
    print("Base model saved to base_model.pth")

if __name__ == "__main__":
    train()
