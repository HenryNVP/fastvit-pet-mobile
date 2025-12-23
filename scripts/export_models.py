from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from timm.models import create_model
import fastvit.models as models  # Register FastViT models with timm
from fastvit.models.modules.mobileone import reparameterize_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained FastViT model to ONNX format."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to FastViT checkpoint.")
    parser.add_argument("--model", type=str, default="fastvit_t8", help="FastViT model name (e.g., fastvit_t8, fastvit_s12).")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes (default: 1000 for ImageNet).")
    parser.add_argument("--input-size", type=int, default=256, help="Input image size (default: 256).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Base directory to place exported files (default: exports).",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Subdirectory name within output-dir (e.g., 'teacher', 'student'). If not specified, files are saved directly in output-dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for export (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17, minimum 14 for ViT models).",
    )
    parser.add_argument("--use-ema", action="store_true", help="Use EMA model weights if available in checkpoint.")
    parser.add_argument("--reparameterize", action="store_true", help="Reparameterize model before export (recommended for inference).")
    parser.add_argument("--fp16", action="store_true", help="Export ONNX model in FP16 (half precision). Weights and activations will be FP16. Default is FP32.")
    return parser.parse_args()


def load_fastvit_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    num_classes: int,
    device: torch.device,
    use_ema: bool = False,
) -> torch.nn.Module:
    """Load a FastViT model from checkpoint."""
    # Create model
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )
    
    # Load checkpoint (weights_only=False for checkpoints that may contain argparse.Namespace)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if use_ema and "model_ema" in checkpoint:
        print("Loading EMA model weights from checkpoint")
        state_dict = checkpoint["model_ema"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Assume checkpoint is the state_dict itself
        state_dict = checkpoint
    
    # Filter out keys with shape mismatches (e.g., classification head with different num_classes)
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    skipped_keys = []
    for key, value in state_dict.items():
        if key in model_state_dict:
            # Check if shapes match
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                skipped_keys.append(f"  - {key} (shape mismatch: {value.shape} -> {model_state_dict[key].shape})")
        else:
            skipped_keys.append(f"  - {key} (not in model)")
    
    # Load with strict=False to handle mismatched keys
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys due to shape mismatch or not in model:")
        for key in skipped_keys[:10]:  # Show first 10
            print(key)
        if len(skipped_keys) > 10:
            print(f"  ... and {len(skipped_keys) - 10} more")
    
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys (first 5: {missing_keys[:5]})")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys (first 5: {unexpected_keys[:5]})")
    
    print(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} weights from checkpoint")
    
    return model


def make_seblock_onnx_compatible(model: torch.nn.Module):
    """Temporarily replace SEBlock forward methods with ONNX-compatible versions.
    
    The original SEBlock uses dynamic kernel sizes from input shape, which ONNX
    cannot handle. This replaces it with adaptive_avg_pool2d which is ONNX-compatible.
    """
    import torch.nn.functional as F
    from fastvit.models.modules.mobileone import SEBlock
    import types
    
    # Store original forward methods
    original_forwards = {}
    
    def onnx_compatible_se_forward(se_block, inputs: torch.Tensor) -> torch.Tensor:
        """ONNX-compatible version using adaptive pooling instead of dynamic kernel size."""
        # Use adaptive_avg_pool2d instead of avg_pool2d with dynamic kernel_size
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = se_block.reduce(x)
        x = F.relu(x)
        x = se_block.expand(x)
        x = torch.sigmoid(x)
        # x is already [B, C, 1, 1] from adaptive pooling, no need to view
        return inputs * x
    
    # Replace SEBlock forward methods
    for name, module in model.named_modules():
        if isinstance(module, SEBlock):
            original_forwards[name] = module.forward
            # Bind the function as a method
            module.forward = types.MethodType(onnx_compatible_se_forward, module)
    
    return original_forwards


def restore_seblock_forwards(model: torch.nn.Module, original_forwards: dict):
    """Restore original SEBlock forward methods."""
    from fastvit.models.modules.mobileone import SEBlock
    
    for name, module in model.named_modules():
        if isinstance(module, SEBlock) and name in original_forwards:
            module.forward = original_forwards[name]


def export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    path: Path,
    opset: int,
    fp16: bool = False,
) -> None:
    """Export model to ONNX format, optionally in FP16 precision.
    
    Args:
        model: Model to export
        dummy_input: Example input tensor
        path: Output path for ONNX model
        opset: ONNX opset version
        fp16: If True, export model in FP16 (half precision), otherwise FP32
    """
    # Check if ONNX is installed
    try:
        import onnx  # type: ignore
    except ImportError:
        raise ImportError(
            "ONNX module is not installed. Please install it with: pip install onnx\n"
            "For ONNX runtime: pip install onnxruntime"
        )
    
    # Make SEBlocks ONNX-compatible by replacing their forward methods
    original_forwards = make_seblock_onnx_compatible(model)
    
    try:
        # Determine export precision
        if fp16:
            # For FP16, convert model to half precision
            print("Converting model to FP16 (half precision)...")
            model_fp16 = model.half()
            dummy_input_fp16 = dummy_input.half()
            export_path = path
            precision_note = "FP16"
        else:
            # FP32 export
            model_fp16 = model
            dummy_input_fp16 = dummy_input
            export_path = path
            precision_note = "FP32"
        
        # Use opset 18 to avoid version conversion issues
        export_opset = max(opset, 18)
        if export_opset > opset:
            print(f"Note: Using opset {export_opset} instead of {opset} to avoid conversion issues")
        
        torch.onnx.export(
            model_fp16,
            dummy_input_fp16,
            export_path,
            export_params=True,
            opset_version=export_opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"Saved {precision_note} ONNX model to {export_path}")
    finally:
        # Restore original forward methods
        restore_seblock_forwards(model, original_forwards)


def main() -> int:
    args = parse_args()
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Model: {args.model}, Input size: {args.input_size}, Num classes: {args.num_classes}")
    print(f"Checkpoint: {args.checkpoint}")
    
    if args.fp16:
        print("FP16 export enabled - model will be exported in half precision (FP16)")
    
    # Load model
    print("\n--- Loading model ---")
    model = load_fastvit_checkpoint(
        args.checkpoint,
        args.model,
        args.num_classes,
        device,
        use_ema=args.use_ema,
    )
    
    # Reparameterize if requested (for faster inference)
    if args.reparameterize:
        print("Reparameterizing model for faster inference...")
        model = reparameterize_model(model)
    
    model = model.to(device)
    model.eval()
    
    # Create output directory (with subdirectory if specified)
    if args.subdir:
        output_dir = args.output_dir / args.subdir
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    input_size = args.input_size
    
    # ONNX export
    print(f"\n--- Exporting ONNX model ---")
    onnx_dummy = torch.randn(1, 3, input_size, input_size, device=device)
    
    # Determine output filename based on precision
    if args.fp16:
        onnx_path = output_dir / "model_fp16.onnx"
    else:
        onnx_path = output_dir / "model.onnx"
    
    try:
        export_onnx(
            model,
            onnx_dummy,
            onnx_path,
            args.onnx_opset,
            fp16=args.fp16,
        )
    except Exception as e:
        print(f"Error exporting ONNX: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nExport complete! Files saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
