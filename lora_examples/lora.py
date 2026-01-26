import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 4, alpha: int = 16):
        """
        Wraps a linear layer with LoRA.
        
        Args:
            original_layer: The original nn.Linear layer to freeze and wrap.
            rank: The rank of the low-rank approximation.
            alpha: The scaling factor.
        """
        super(LoRALinear, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # LoRA weights
        # A: (rank, in_features) - initialized with Kaiming uniform
        # B: (out_features, rank) - initialized to zero
        self.lora_A = nn.Parameter(torch.Tensor(rank, original_layer.in_features))
        self.lora_B = nn.Parameter(torch.Tensor(original_layer.out_features, rank))

        # Output reset
        self.reset_parameters()

        self.merged = False
    
    def reset_parameters(self):
        # Initialize A with random gaussian, B with zero
        # This ensures that at start, LoRA output is 0, so behavior matches original model
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Standard forward: Result = Wx + (BAx * scale)
        original_out = self.original_layer(x)
        
        if self.merged:
            # If weights are already merged, original_layer contains (W + BA)
            return original_out
        
        # Compute LoRA branch: x @ A^T @ B^T * scaling
        # (batch, in) @ (in, rank) -> (batch, rank)
        # (batch, rank) @ (rank, out) -> (batch, out)
        lora_out = (x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        
        return original_out + lora_out

    def merge(self):
        """
        Merges the LoRA weights into the original weights.
        W_new = W_old + B @ A * scaling
        """
        if self.merged:
            return

        # Weight shape is (out, in)
        # B @ A gives (out, rank) @ (rank, in) -> (out, in)
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.original_layer.weight.data += delta_w
        self.merged = True

    def unmerge(self):
        """
        Removes LORA influence from original weights.
        """
        if not self.merged:
            return
            
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.original_layer.weight.data -= delta_w
        self.merged = False


def apply_lora(model, target_modules=None, rank=4, alpha=16):
    """
    Replaces specified linear layers in the model with LoRALinear layers.
    If target_modules is None, replaces all linear layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # If target_modules is specified, check if this module name is in it
            # For simplicity in this example, we might just replace all linears or specific ones.
            # Here we replace all nn.Linear if target_modules is None
            if target_modules is None or name in target_modules:
                # Replace with LoRA
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(model, name, lora_layer)
        else:
            # Recursively apply
            apply_lora(module, target_modules, rank, alpha)

def get_lora_params(model):
    """Returns only LoRA parameters for optimization."""
    params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            params.append(param)
    return params
