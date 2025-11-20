from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from timm.models import create_model
import fastvit.models as models  # Register FastViT models with timm
from fastvit.models.modules.mobileone import reparameterize_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference speed across model formats for FastViT models.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to FastViT checkpoint.")
    parser.add_argument("--model", type=str, default="fastvit_t8", help="FastViT model name (e.g., fastvit_t8, fastvit_s12).")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes (default: 1000 for ImageNet).")
    parser.add_argument("--input-size", type=int, default=256, help="Input image size (default: 256).")
    parser.add_argument("--torchscript", type=Path, default=None, help="Path to TorchScript model.")
    parser.add_argument("--onnx", type=Path, default=None, help="Path to ONNX model.")
    parser.add_argument("--exports-dir", type=Path, default=Path("exports"), help="Base directory where exported models are stored (default: exports).")
    parser.add_argument("--subdir", type=str, default=None, help="Subdirectory within exports-dir (e.g., 'teacher', 'student'). If not specified, looks directly in exports-dir.")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available, else cpu).")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--runs", type=int, default=100, help="Timed iterations.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA model weights if available in checkpoint.")
    parser.add_argument("--reparameterize", action="store_true", help="Reparameterize model for faster inference (recommended).")
    return parser.parse_args()


def benchmark_pytorch(model: torch.nn.Module, dummy: torch.Tensor, warmup: int, runs: int) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if torch.cuda.is_available() and dummy.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            model(dummy)
        if torch.cuda.is_available() and dummy.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
    fps = runs * dummy.size(0) / elapsed
    latency = elapsed / runs
    return {"fps": fps, "latency": latency, "total_time": elapsed}


def benchmark_torchscript(path: Path, dummy: torch.Tensor, device: torch.device, warmup: int, runs: int) -> Dict[str, float]:
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return benchmark_pytorch(model, dummy, warmup, runs)


def benchmark_onnx(path: Path, dummy: np.ndarray, warmup: int, runs: int) -> Dict[str, float]:
    sess = ort.InferenceSession(path.as_posix(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inputs = {sess.get_inputs()[0].name: dummy}
    for _ in range(warmup):
        sess.run(None, inputs)
    start = time.time()
    for _ in range(runs):
        sess.run(None, inputs)
    elapsed = time.time() - start
    fps = runs * dummy.shape[0] / elapsed
    latency = elapsed / runs
    return {"fps": fps, "latency": latency, "total_time": elapsed}


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
    
    results: Dict[str, Dict[str, float]] = {}
    input_size = args.input_size

    # Load and benchmark PyTorch model
    print("\n--- Loading PyTorch model ---")
    pytorch_model = load_fastvit_checkpoint(
        args.checkpoint,
        args.model,
        args.num_classes,
        device,
        use_ema=args.use_ema,
    )
    
    # Reparameterize if requested (for faster inference)
    if args.reparameterize:
        print("Reparameterizing model for faster inference...")
        pytorch_model = reparameterize_model(pytorch_model)
    
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()
    
    dummy = torch.randn(args.batch_size, 3, input_size, input_size, device=device)
    print(f"Benchmarking PyTorch model (warmup={args.warmup}, runs={args.runs})...")
    results["pytorch"] = benchmark_pytorch(pytorch_model, dummy, args.warmup, args.runs)

    # Determine default export paths
    if args.subdir:
        exports_base = ROOT_DIR / args.exports_dir / args.subdir
    else:
        exports_base = ROOT_DIR / args.exports_dir
    
    # TorchScript
    if args.torchscript:
        ts_path = args.torchscript
    else:
        ts_path = exports_base / "model_scripted.pt"
    
    if ts_path.exists():
        print(f"\n--- Benchmarking TorchScript model: {ts_path} ---")
        results["torchscript"] = benchmark_torchscript(ts_path, dummy, device, args.warmup, args.runs)
    else:
        print(f"\n[warn] TorchScript model not found at {ts_path}. Skipping.")
        print(f"      (Checked: {ts_path.absolute()})")

    # ONNX (expects same input size as model)
    if args.onnx:
        onnx_path = args.onnx
    else:
        onnx_path = exports_base / "model.onnx"
    
    if onnx_path.exists():
        print(f"\n--- Benchmarking ONNX model: {onnx_path} ---")
        dummy_onnx = torch.randn(args.batch_size, 3, input_size, input_size).numpy()
        results["onnx"] = benchmark_onnx(onnx_path, dummy_onnx, args.warmup, args.runs)
    else:
        print(f"\n[warn] ONNX model not found at {onnx_path}. Skipping.")
        print(f"      (Checked: {onnx_path.absolute()})")

    print("\n" + "="*70)
    print("--- Benchmark Results ---")
    print("="*70)
    print(f"Batch size: {args.batch_size}, Runs: {args.runs}, Warmup: {args.warmup}")
    print("-"*70)
    for name, metrics in results.items():
        # Latency is per batch, so time per image = latency / batch_size
        time_per_image_ms = (metrics['latency'] * 1000) / args.batch_size
        print(f"{name:15s}: FPS={metrics['fps']:8.2f}  "
              f"Latency/batch={metrics['latency']*1000:6.2f}ms  "
              f"Time/image={time_per_image_ms:6.2f}ms  "
              f"Total={metrics['total_time']:.3f}s")
    print("="*70)
    print(f"\nNote: Time per image = Latency per batch / batch size")
    if args.batch_size == 1:
        print("      (With batch_size=1, latency equals time per image)")
    else:
        print(f"      (With batch_size={args.batch_size}, processing {args.batch_size} images per batch)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
