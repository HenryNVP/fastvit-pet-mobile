from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import load_configs, to_namespace
from src.models import create_model
from src.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained model to TorchScript and ONNX formats."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/regnety_016.yaml"),
        help="Model configuration file.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Training configuration file.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Data configuration file (used for consistency).",
    )
    parser.add_argument(
        "--aug-config",
        type=Path,
        default=Path("configs/aug.yaml"),
        help="Augmentation configuration file (used to infer input size).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Directory to place exported files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for export (defaults to preferred device or CPU).",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17, minimum 14 for ViT models).",
    )
    return parser.parse_args()


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def infer_sizes(augment_cfg: Dict[str, Any] | None, fallback: int = 224) -> Tuple[int, int]:
    input_size = fallback
    crop_size = fallback

    if augment_cfg:
        size = augment_cfg.get("size")
        if isinstance(size, (int, float)):
            input_size = int(size)

        train_cfg = augment_cfg.get("train", {})
        random_crop = train_cfg.get("random_resized_crop")
        if isinstance(random_crop, dict):
            crop_size = int(random_crop.get("size", crop_size))
        elif isinstance(random_crop, (int, float)) and not isinstance(random_crop, bool):
            crop_size = int(random_crop)

        eval_cfg = augment_cfg.get("eval", {})
        center_crop = eval_cfg.get("center_crop")
        if isinstance(center_crop, (int, float)) and not isinstance(center_crop, bool):
            crop_size = int(center_crop)

    crop_size = max(1, crop_size)
    return input_size, crop_size


class CenterCropModule(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = int(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        th = tw = self.size
        h, w = x.shape[-2:]
        th = min(th, h)
        tw = min(tw, w)
        i = (h - th) // 2
        j = (w - tw) // 2
        return x[..., i : i + th, j : j + tw]


def export_torchscript(model: torch.nn.Module, dummy_input: torch.Tensor, path: Path) -> None:
    try:
        scripted = torch.jit.script(model)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"[warn] torch.jit.script failed ({exc}); falling back to trace.")
        scripted = torch.jit.trace(model, dummy_input)
    scripted.save(path)
    print(f"Saved TorchScript model to {path}")


def to_torchscript(
    checkpoint: Path,
    output_path: Path,
    model_cfg: Any,
    device: torch.device,
    input_size: int,
    crop_size: int,
) -> None:
    model = create_model(model_cfg).to(device)
    load_checkpoint_weights(model, checkpoint, device)
    model.eval()

    export_model = torch.nn.Sequential(CenterCropModule(crop_size), model)
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    export_torchscript(export_model, dummy_input, output_path)


def export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    path: Path,
    opset: int,
) -> None:
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Saved ONNX model to {path}")


def main() -> int:
    args = parse_args()
    config = load_configs(
        [
            args.train_config,
            args.data_config,
            args.model_config,
            args.aug_config,
        ]
    )

    model_cfg = to_namespace(config["model"])
    augment_cfg = config.get("augment")

    input_size, crop_size = infer_sizes(augment_cfg, fallback=224)
    device_name = args.device or get_device(config.get("train", {}).get("device_preference"))
    device = torch.device(device_name)

    model = create_model(model_cfg).to(device)
    load_checkpoint_weights(model, args.checkpoint, device)
    model.eval()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # TorchScript export (includes center crop wrapper)
    ts_model = torch.nn.Sequential(CenterCropModule(crop_size), model)
    ts_dummy = torch.randn(1, 3, input_size, input_size, device=device)
    ts_path = output_dir / "model_scripted.pt"
    export_torchscript(ts_model, ts_dummy, ts_path)

    # ONNX export expects already-cropped inputs
    onnx_dummy = torch.randn(1, 3, crop_size, crop_size, device=device)
    onnx_path = output_dir / "model.onnx"
    # Use configurable opset version (default 17, ViT requires >=14 for scaled_dot_product_attention)
    export_onnx(model, onnx_dummy, onnx_path, args.onnx_opset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
