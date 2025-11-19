from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import load_configs, to_namespace
from src.models import create_model
from src.utils import get_device
from scripts.export_models import (
    infer_sizes,
    load_checkpoint_weights,
    CenterCropModule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference speed across model formats.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/regnety_016.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--aug-config", type=Path, default=Path("configs/aug.yaml"))
    parser.add_argument("--torchscript", type=Path, default=None, help="Path to TorchScript model.")
    parser.add_argument("--onnx", type=Path, default=None, help="Path to ONNX model.")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto).")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--runs", type=int, default=100, help="Timed iterations.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
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


def main() -> int:
    args = parse_args()
    config = load_configs([args.train_config, args.data_config, args.model_config, args.aug_config])
    model_cfg = to_namespace(config["model"])
    augment_cfg = config.get("augment")

    input_size, crop_size = infer_sizes(augment_cfg, fallback=224)
    device_name = args.device or get_device(config.get("train", {}).get("device_preference"))
    device = torch.device(device_name)

    results: Dict[str, Dict[str, float]] = {}

    # PyTorch model
    pytorch_model = create_model(model_cfg).to(device)
    load_checkpoint_weights(pytorch_model, args.checkpoint, device)
    pytorch_model.eval()
    dummy = torch.randn(args.batch_size, 3, input_size, input_size, device=device)
    results["pytorch"] = benchmark_pytorch(pytorch_model, dummy, args.warmup, args.runs)

    # TorchScript
    ts_path = args.torchscript or (Path("exports") / "model_scripted.pt")
    if ts_path.exists():
        results["torchscript"] = benchmark_torchscript(ts_path, dummy, device, args.warmup, args.runs)
    else:
        print(f"[warn] TorchScript model not found at {ts_path}. Skipping.")

    # ONNX (expects cropped input)
    onnx_path = args.onnx or (Path("exports") / "model.onnx")
    if onnx_path.exists():
        dummy_onnx = torch.randn(args.batch_size, 3, crop_size, crop_size).numpy()
        results["onnx"] = benchmark_onnx(onnx_path, dummy_onnx, args.warmup, args.runs)
    else:
        print(f"[warn] ONNX model not found at {onnx_path}. Skipping.")

    print("--- Benchmark Results ---")
    for name, metrics in results.items():
        print(f"{name}: FPS={metrics['fps']:.2f} Latency={metrics['latency']*1000:.2f}ms Total={metrics['total_time']:.3f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
