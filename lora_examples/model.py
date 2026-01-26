import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

if __name__ == "__main__":
    model = SimpleModel()
    print(model)
    dummy_input = torch.randn(1, 10)
    output = model(dummy_input)
    print("Output shape:", output.shape)
