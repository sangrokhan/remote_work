
import torch
import sys
import os

# Add the parent directory to sys.path to resolve imports if run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from transfer_learning.transfer_utils import load_base_model, modify_input_layer, modify_output_layer

def run_demo():
    print("--- Transfer Learning Demo ---")
    
    # 1. Load Base Model
    print("1. Loading base ResNet18 model...")
    model = load_base_model(pretrained=True)
    print("   Model loaded.")
    
    # Check original first layer
    print(f"   Original first conv layer: {model.conv1}")
    
    # 2. Modify Input Layer (e.g., for RGBA images -> 4 channels)
    target_channels = 4
    print(f"2. Modifying input layer to accept {target_channels} channels...")
    model = modify_input_layer(model, in_channels=target_channels)
    print(f"   Modified first conv layer: {model.conv1}")
    
    # 3. Modify Output Layer (e.g., for 10 classes)
    target_classes = 10
    print(f"3. Modifying output layer to output {target_classes} classes...")
    model = modify_output_layer(model, num_classes=target_classes)
    print(f"   Modified final fc layer: {model.fc}")
    
    # 4. Verify with Dummy Input
    print("4. Verifying with dummy input...")
    batch_size = 2
    # Create random input: (batch_size, channels, height, width)
    dummy_input = torch.randn(batch_size, target_channels, 224, 224)
    
    # Forward pass
    try:
        output = model(dummy_input)
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        if output.shape == (batch_size, target_classes):
            print("   SUCCESS: Output shape matches expected dimensions.")
        else:
            print(f"   FAILURE: Expected output shape {(batch_size, target_classes)}, got {output.shape}")
            
    except Exception as e:
        print(f"   FAILURE: Error during forward pass: {e}")

    # 5. Verify Class Incremental Weight Preservation
    print("\n5. Verifying Class Incremental Learning (Weight Preservation)...")
    
    # Start with a 5-class model
    inc_model = load_base_model()
    inc_model = modify_output_layer(inc_model, num_classes=5)
    
    # Capture weights for standard class (e.g., class 0)
    original_weights = inc_model.fc.weight.data.clone()
    print(f"   Original 5-class weights shape: {original_weights.shape}")
    
    # Increment to 6 classes
    print("   Incrementing to 6 classes...")
    inc_model = modify_output_layer(inc_model, num_classes=6)
    new_weights = inc_model.fc.weight.data
    print(f"   New 6-class weights shape: {new_weights.shape}")
    
    # Check if first 5 rows are preserved
    if torch.allclose(original_weights, new_weights[:5, :]):
        print("   SUCCESS: Weights for original 5 classes are preserved!")
    else:
        print("   FAILURE: Weights were not preserved correctly.")
        
    # Check if new row is initialized (not zero, random) and likely different from others (not a strict check but good sanity)
    # Just ensuring it exists and has correct shape
    if new_weights.shape[0] == 6:
        print("   SUCCESS: New layer has correct output count.")

if __name__ == "__main__":
    run_demo()
