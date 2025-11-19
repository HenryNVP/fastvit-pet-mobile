"""
Example script showing how to load a finetuned FastViT model.

This demonstrates multiple ways to load checkpoints saved during training.
"""
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from timm.models import create_model, load_checkpoint
import fastvit.models as models
from fastvit.models.modules.mobileone import reparameterize_model


def load_finetuned_model_method1(checkpoint_path, model_name="fastvit_t8", num_classes=37, device="cuda"):
    """
    Method 1: Using timm's load_checkpoint (recommended)
    
    This handles checkpoint format automatically and works with checkpoints
    saved by the training script.
    """
    # Create model with correct number of classes
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )
    
    # Load checkpoint (handles "model", "state_dict", or direct state_dict)
    load_checkpoint(model, checkpoint_path, use_ema=False)
    
    # Set to eval mode
    model.eval()
    
    # Reparameterize for inference (optional, for faster inference)
    model = reparameterize_model(model)
    
    # Move to device
    model = model.to(device)
    
    return model


def load_finetuned_model_method2(checkpoint_path, model_name="fastvit_t8", num_classes=37, device="cuda"):
    """
    Method 2: Manual loading with format detection
    
    Useful when you need more control over the loading process.
    """
    # Create model
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Assume checkpoint is the state_dict itself
        state_dict = checkpoint
    
    # Load with strict=False to handle mismatched keys (e.g., different num_classes)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")
    
    # Set to eval mode
    model.eval()
    
    # Reparameterize for inference (optional)
    model = reparameterize_model(model)
    
    # Move to device
    model = model.to(device)
    
    return model


def load_finetuned_model_method3(checkpoint_path, model_name="fastvit_t8", num_classes=37, device="cuda", use_ema=False):
    """
    Method 3: Load EMA model if available
    
    EMA (Exponential Moving Average) models often perform better.
    """
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load EMA model first if requested
    if use_ema and "model_ema" in checkpoint:
        print("Loading EMA model weights")
        state_dict = checkpoint["model_ema"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = reparameterize_model(model)
    model = model.to(device)
    
    return model


def example_usage():
    """Example of how to use the loaded model for inference"""
    
    # Path to your finetuned checkpoint
    checkpoint_path = "./output/train/20240101-120000-fastvit_t8-256/model_best.pth.tar"
    
    # Load model
    model = load_finetuned_model_method1(
        checkpoint_path=checkpoint_path,
        model_name="fastvit_t8",
        num_classes=37,  # Your number of classes
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Example inference
    model.eval()
    with torch.no_grad():
        # Create dummy input (batch_size=1, channels=3, height=256, width=256)
        dummy_input = torch.randn(1, 3, 256, 256).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Forward pass
        output = model(dummy_input)
        
        # Get predictions
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        
        print(f"Predicted class: {predicted_class.item()}")
        print(f"Confidence: {probabilities[0][predicted_class].item():.4f}")


if __name__ == "__main__":
    example_usage()

