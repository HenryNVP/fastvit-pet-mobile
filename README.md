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

On-device performance on Android (Oxford-IIIT Pet dataset, 37 classes):

| Model / Configuration | Avg Inference Time | Top-1 Accuracy | Top-5 Accuracy |
|----------------------|-------------------|----------------|---------------|
| sa12_fp16 | 40.61 ms | 90.15% | 99.13% |
| sa12_fp32 | 190.23 ms | 90.15% | 99.13% |
| sa12P_fp16 | 39.82 ms | 90.15% | 99.13% |
| sa12P_fp32 | 195.01 ms | 90.15% | 99.13% |

**Key Findings:** This project enhances FastViT for mobile deployment by integrating **Performer attention** (reducing attention complexity from $O(N^2)$ to $O(N)$) and knowledge distillation.The distilled FastViT SA12_P model (quantized to FP16) achieves the lowest latency of 39.82 ms, outperforming the baseline FastViT SA12 FP16 (40.61 ms). This configuration restores the expected performance gains from quantization, where the FP16 model is significantly faster (~4.9x) than its FP32 counterpart. While the latency reduction compared to the baseline is modest (~0.8 ms), the SA12_P model maintains an identical Top-1 accuracy of 90.15%, demonstrating that Performer attention can be effectively quantized for mobile edge devices without accuracy loss.

## Quick Start

### 1. Prepare data
```bash
python -m scripts.get_data
python -m scripts.split_dataset
python -m scripts.create_test_app
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
python -m fastvit.train data \
    --model fastvit_sa12_P \
    --resume checkpoint.pth.tar \
    --finetune \
    --freeze-stages 0 1 2 \
    --num-classes 37
```

The `--freeze-stages` argument accepts stage indices (0-indexed). Classification head is always trainable.

## Export models

```bash
python -m fastvit.export_model
python -m scripts.export_models
```


## License

This project is licensed under MIT. However, the `fastvit/` directory contains code from [Apple's FastViT](https://github.com/apple/ml-fastvit), which is licensed under Apple's proprietary license. See [`fastvit/LICENSE`](fastvit/LICENSE) for details.

