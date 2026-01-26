
import torch
import torch.nn as nn
from .local_model import SimpleCNN

def load_base_model(model_name: str = 'local_simple_cnn', pretrained: bool = False):
    """
    Loads a base model.
    Previously loaded from torchvision, now utilizes a local SimpleCNN definition
    to avoid external network dependencies.
    
    Args:
        model_name (str): Name of the model (default: 'local_simple_cnn').
        pretrained (bool): Ignored for local model, kept for API compatibility.
        
    Returns:
        model: The loaded PyTorch model.
    """
    # Simply instantiate the local model
    model = SimpleCNN()
    return model

def modify_output_layer(model: nn.Module, num_classes: int):
    """
    Modifies the last fully connected layer of the model to match num_classes.
    If num_classes > old_classes, it transfers the existing weights and biases.
    
    Args:
        model (nn.Module): The PyTorch model.
        num_classes (int): The number of output classes.
        
    Returns:
        model: The modified model.
    """
    # Assuming ResNet architecture where the last layer is 'fc'
    if hasattr(model, 'fc'):
        old_fc = model.fc
        in_features = old_fc.in_features
        old_classes = old_fc.out_features
        
        new_fc = nn.Linear(in_features, num_classes)
        
        # If we are increasing the number of classes, we might want to keep the old ones.
        # This is useful for class-incremental learning.
        if num_classes > old_classes:
            with torch.no_grad():
                # Copy weights: new_fc has shape (num_classes, in_features)
                new_fc.weight[:old_classes, :] = old_fc.weight
                if old_fc.bias is not None and new_fc.bias is not None:
                    new_fc.bias[:old_classes] = old_fc.bias
                    
        model.fc = new_fc
    else:
        raise NotImplementedError("This utility currently only supports models with a 'fc' final layer (like ResNet).")
    
    return model

def modify_input_layer(model: nn.Module, in_channels: int = 3):
    """
    Modifies the first convolutional layer to accept a different number of input channels.
    It attempts to reuse existing weights by averaging or copying them.
    
    Args:
        model (nn.Module): The PyTorch model.
        in_channels (int): New number of input channels.
        
    Returns:
        model: The modified model.
    """
    # Assuming ResNet architecture where the first layer is 'conv1'
    if hasattr(model, 'conv1'):
        old_conv = model.conv1
        
        # Check if modification is actually needed
        if old_conv.in_channels == in_channels:
            return model
            
        # Create new conv layer with same parameters but different in_channels
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Initialize weights
        with torch.no_grad():
            if in_channels > 3:
                # Case: More channels than original (e.g. RGBA).
                # Copy original weights to first 3 channels
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # Initialize remaining channels (e.g. by averaging original weights or random)
                # Here we use the mean of the original weights to initialize extra channels
                # This is a common heuristic to keep activation scales somewhat similar
                mean_weight = torch.mean(old_conv.weight, dim=1, keepdim=True)
                for i in range(3, in_channels):
                    new_conv.weight[:, i:i+1, :, :] = mean_weight
            elif in_channels < 3:
                # Case: Fewer channels (e.g. Grayscale).
                # Average original weights across RGB channels
                new_conv.weight = nn.Parameter(torch.mean(old_conv.weight, dim=1, keepdim=True))
                # If target is not exactly 1 channel but still < 3 (rare), we might need different logic,
                # but for in_channels=1 this works well.
            
            # Copy bias if it exists
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
                
        model.conv1 = new_conv
    else:
        raise NotImplementedError("This utility currently only supports models with a 'conv1' first layer (like ResNet).")
        
    return model
