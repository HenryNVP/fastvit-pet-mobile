FastViT Pet Mobile
===================

End‑to‑end setup for training and deploying **FastViT** models for pet classification on Android devices. This repository extends [Apple's FastViT](https://github.com/apple/ml-fastvit) with custom features for mobile deployment and transfer learning. The project integrates **Performer attention** to reduce attention complexity from $O(N^2)$ to $O(N)$, **knowledge distillation** to transfer knowledge from teacher models, and **quantization** (FP16/FP32) to optimize inference speed on mobile devices while maintaining accuracy.

**What's included:**
- FastViT model implementations (original + Performer attention variants)
- Training pipeline with stage freezing for transfer learning
- Data preparation scripts for Oxford-IIIT Pet dataset
- Model export to ONNX for mobile deployment
- Android app (`Petclassify`) for on-device inference and evaluation
- Log analysis tools for accuracy and latency metrics

**Custom features:**
- **Performer attention models** (`fastvit_sa12_P`) - Linear attention variants for improved efficiency
- **Stage freezing** (`--freeze-stages`) - Freeze early layers during transfer learning

## Benchmarks

On-device performance on Android Legion Phone Duel 2 (Oxford-IIIT Pet dataset, 37 classes):

| Model / Configuration | Avg Inference Time | Top-1 Accuracy | Top-5 Accuracy |
|----------------------|-------------------|----------------|---------------|
| sa12_fp16 | 39.22 ms | 90.15% | 99.13% |
| sa12_fp32 | 190.23 ms | 90.15% | 99.13% |
| sa12P_fp16 | 39.50 ms | 90.15% | 99.13% |
| sa12P_fp32 | 189.99 ms | 90.15% | 99.13% |

**Key Findings:** This project evaluates FastViT for mobile deployment by comparing standard Multi-Head Self-Attention (MHSA) against **Performer attention** (which theoretically reduces complexity from $O(N^2)$ to $O(N)$).The results indicate that Performer attention performs almost identically to MHSA on this architecture. This suggests that for the short sequence lengths ($N$) typical in these model stages, the asymptotic advantage of linear attention does not translate into a practical reduction in inference time compared to standard attention.Despite the lack of speedup from the architectural change, both models demonstrate excellent quantization efficiency, with FP16 configurations running approximately 4.8x faster than their FP32 counterparts while maintaining an identical Top-1 accuracy.

## Quick Start

### 1. Prepare data
```bash
python scripts/get_data.py
python scripts/split_dataset.py
```

### 2. Prepare model
- Prepare model `fasvit.onnx` and `model.onnx.data` in `Petclassify/app/src/main/assets/`

### 3. Transfer test data to device
```bash
adb push data/test_app /storage/emulated/0/Download/
```

### 4. Run app
- Open `Petclassify/` in Android Studio
- Run on device/emulator
- Grant "All files access" permission when prompted
- Analyze results in Log Cat


## Custom Features

### Performer Attention Models
Linear attention variants using `performer-pytorch` for improved efficiency:
- `fastvit_sa12_P` - Performer variant of SA12

### Stage Freezing for Transfer Learning
Freeze early model stages while training classification head:

```bash
python fastvit/train.py data \
    --model fastvit_sa12 \
    --resume fastvit_sa12.pth.tar \
    --finetune \
    --freeze-stages 0 1 2 \
    --num-classes 37 \
    -b 192 --lr 1e-3 \
    --epochs 30 \
    --native-amp \
    --no-model-ema \
    --output ./output \
    --input-size 3 256 256 \
    --drop-path 0.1
```

The `--freeze-stages` argument accepts stage indices (0-indexed). Classification head is always trainable.

### Knowledge Distillation
Train Performer attention models with knowledge distillation from a teacher model:

```bash
python fastvit/train.py data \
    --model fastvit_sa12_P \
    --initial-checkpoint checkpoint.pth.tar \
    --finetune \
    --num-classes 37 \
    -b 128 \
    --lr 1e-4 \
    --clip-grad 1.0 \
    --epochs 20 \
    --output ./output \
    --input-size 3 256 256 \
    --teacher-model fastvit_sa12 \
    --teacher-path teacher_checkpoint.pth.tar \
    --distillation-type soft \
    --distillation-tau 3.0 \
    --distillation-alpha 0.5 \
    --no-model-ema \
    --drop-path 0.1
```

## Export models

Export models to ONNX format for mobile deployment:

```bash
# Export FP32 model
python scripts/export_models.py \
    --checkpoint checkpoint.pth.tar \
    --model fastvit_sa12 \
    --num-classes 37 \
    --input-size 256 \
    --reparameterize \
    --output-dir exports/fastvit_sa12-fp32

# Export FP16 quantized model
python scripts/export_models.py \
    --checkpoint checkpoint.pth.tar \
    --model fastvit_sa12 \
    --num-classes 37 \
    --input-size 256 \
    --reparameterize \
    --fp16 \
    --output-dir exports/fastvit_sa12_fp16
```


## License

This project is licensed under MIT. However, the `fastvit/` directory contains code from [Apple's FastViT](https://github.com/apple/ml-fastvit), which is licensed under Apple's proprietary license. See [`fastvit/LICENSE`](fastvit/LICENSE) for details.

