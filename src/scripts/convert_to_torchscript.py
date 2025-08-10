import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.classifier.model import get_model

def convert_to_torchscript(model_path, output_path, num_classes=5, input_size=224, trace_mode=True):
    """
    Convert a PyTorch model to TorchScript format for deployment.
    
    Args:
        model_path (str): Path to the saved PyTorch model (.pth file)
        output_path (str): Path where the TorchScript model will be saved
        num_classes (int): Number of classes in the model
        input_size (int): Input image size (assumes square images)
        trace_mode (bool): If True, use torch.jit.trace, else use torch.jit.script
    """
    
    # Device - use CPU for better compatibility with Raspberry Pi
    device = torch.device("cpu")
    
    print(f"Loading model from: {model_path}")
    
    # Load the model architecture
    model = get_model(num_classes=num_classes)
    
    # Load the trained weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✓ Model weights loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model weights: {e}")
        return False
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    try:
        if trace_mode:
            # Create a dummy input tensor for tracing
            dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
            
            print("Converting model using torch.jit.trace...")
            # Trace the model
            with torch.no_grad():
                traced_model = torch.jit.trace(model, dummy_input)
            
            # Verify the traced model works
            print("Verifying traced model...")
            original_output = model(dummy_input)
            traced_output = traced_model(dummy_input)
            
            # Check if outputs are close
            if torch.allclose(original_output, traced_output, atol=1e-5):
                print("✓ Model tracing verification successful")
            else:
                print("⚠ Warning: Traced model outputs differ from original")
                
        else:
            print("Converting model using torch.jit.script...")
            traced_model = torch.jit.script(model)
    
    except Exception as e:
        print(f"✗ Error during model conversion: {e}")
        if trace_mode:
            print("Trying with script mode instead...")
            try:
                traced_model = torch.jit.script(model)
                print("✓ Script mode conversion successful")
            except Exception as e2:
                print(f"✗ Script mode also failed: {e2}")
                return False
        else:
            return False
    
    # Save the TorchScript model
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        traced_model.save(output_path)
        print(f"✓ TorchScript model saved to: {output_path}")
        
        # Check file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        print(f"✓ Model size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Error saving TorchScript model: {e}")
        return False

def verify_torchscript_model(model_path, input_size=224):
    """
    Verify that the saved TorchScript model can be loaded and used for inference.
    """
    try:
        print(f"\nVerifying TorchScript model: {model_path}")
        
        # Load the TorchScript model
        loaded_model = torch.jit.load(model_path, map_location='cpu')
        loaded_model.eval()
        
        # Create a test input
        test_input = torch.randn(1, 3, input_size, input_size)
        
        # Run inference
        with torch.no_grad():
            output = loaded_model(test_input)
            
        print(f"✓ Model loaded successfully")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Model is ready for deployment!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TorchScript')
    parser.add_argument('--model_path', type=str, 
                       default='saved_models/best_model.pth',
                       help='Path to the PyTorch model file')
    parser.add_argument('--output_path', type=str, 
                       default='saved_models/best_model_torchscript.pt',
                       help='Output path for TorchScript model')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes in the model')
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size (height and width)')
    parser.add_argument('--script_mode', action='store_true',
                       help='Use torch.jit.script instead of torch.jit.trace')
    parser.add_argument('--verify', action='store_true', default=True,
                       help='Verify the converted model')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    model_path = os.path.abspath(args.model_path)
    output_path = os.path.abspath(args.output_path)
    
    # Check if input model exists
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return
    
    print("=" * 60)
    print("PyTorch to TorchScript Converter")
    print("=" * 60)
    print(f"Input model: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Input size: {args.input_size}x{args.input_size}")
    print(f"Conversion mode: {'script' if args.script_mode else 'trace'}")
    print("=" * 60)
    
    # Convert the model
    success = convert_to_torchscript(
        model_path=model_path,
        output_path=output_path,
        num_classes=args.num_classes,
        input_size=args.input_size,
        trace_mode=not args.script_mode
    )
    
    if success and args.verify:
        verify_torchscript_model(output_path, args.input_size)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ CONVERSION FAILED")
        print("=" * 60)

if __name__ == "__main__":
    main()
