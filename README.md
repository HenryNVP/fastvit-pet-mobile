FastViT Pet Mobile
===================

End‑to‑end setup for training and deploying **FastViT** models for pet classification on Android devices. This repository extends [Apple's FastViT](https://github.com/apple/ml-fastvit) with custom features for mobile deployment and transfer learning.
Demo video of the benchmark app is available [here](https://drive.google.com/drive/folders/1UZZX45Nn0P_neb1rYmjt9dAVj5ZWijFA?usp=drive_link).

**What's included:**
- FastViT model implementations (original + Performer attention variants)
- Training pipeline with stage freezing for transfer learning
- Data preparation scripts for Oxford-IIIT Pet dataset
- Model export to ONNX for mobile deployment
- Android app (`Petclassify`) for on-device inference and evaluation
- Log analysis tools for accuracy and latency metrics

**Custom features:**
- **Performer attention models** (`fastvit_t8_P`, `fastvit_sa12_P`) - Linear attention variants for improved efficiency
- **Stage freezing** (`--freeze-stages`) - Freeze early layers during transfer learning

## Benchmarks

On-device performance on Android (Oxford-IIIT Pet dataset, 37 classes):

| Model | Avg Inference Time | Average FPS | Micro Top-1 | Micro Top-5 | Macro Top-1 | Macro Top-5 |
|-------|-------------------|-------------|-------------|-------------|-------------|-------------|
| FastViT T8 | 48.75 ms | 20.5 | 90.15% | 99.13% | 90.12% | 99.14% |
| FastViT SA12_P (distilled) | 43.43 ms | 23.0 | 90.10% | 99.12% | 90.18% | 99.14% |

**Key Findings:** This project enhances FastViT for mobile deployment by integrating **Performer attention** (reducing attention complexity from O(N²) to O(N)) and **knowledge distillation**. The distilled FastViT SA12_P model achieves comparable accuracy to FastViT T8 while reducing latency by ~5ms and increasing FPS by ~2.5, despite having more parameters. These optimizations demonstrate that attention mechanism improvements and distillation can accelerate larger ViT-based models for real-time mobile and edge deployment.

## Quick Start

### 1. Prepare data
```bash
python -m scripts.get_data
python -m scripts.split_dataset
python -m scripts.create_test_app
```

### 2. Prepare model
- Copy `model/t8/fastvit.onnx` to `Petclassify/app/src/main/assets/fastvit.onnx`

### 3. Transfer test data to device
```bash
adb push data/test_app /storage/emulated/0/Download/
```

### 4. Run app
- Open `Petclassify/` in Android Studio
- Run on device/emulator
- Grant "All files access" permission when prompted

### 5. Analyze results
```bash
adb logcat -s EVAL_RESULT > results.txt
python -m scripts.analyze_logs results.txt
```


## Custom Features

### Performer Attention Models
Linear attention variants using `performer-pytorch` for improved efficiency:
- `fastvit_t8_P` - Performer variant of T8
- `fastvit_sa12_P` - Performer variant of SA12

**Usage:**
```bash
pip install performer-pytorch
python -m fastvit.train data --model fastvit_t8_P --num-classes 37
```

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

