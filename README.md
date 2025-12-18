FastViT Pet Mobile
===================

This repository contains a small end‑to‑end setup for running **FastViT** pet classification on desktop and Android:

- Prepare a train/val/test split from the Oxford‑IIIT Pet dataset.
- Export FastViT models to ONNX for mobile.
- Run the Android app (`Petclassify`) to classify pet images.
- Collect and analyze on‑device evaluation logs.

Demo video is available [here](https://drive.google.com/drive/folders/1UZZX45Nn0P_neb1rYmjt9dAVj5ZWijFA?usp=sharing).

## Data preparation

1. **Download raw data**
   - From repo root:

   ```bash
   python -m scripts.get_data
   ```

   This populates `data/raw` with:
   - `images/`
   - `annotations/` (including `list.txt`, `trainval.txt`, `test.txt`)

2. **Create train / val / test splits**

   ```bash
   python -m scripts.split_dataset
   ```

   This creates:
   - `data/train/`, `data/validation/`, `data/test/` with resized \(256 × 256\) JPEGs
   - CSV label files:
     - `data/train_labels.csv`
     - `data/val_labels.csv`
     - `data/test_labels.csv`

   Split logic:
   - Train: all images from `annotations/trainval.txt`
   - Val/Test: images from `annotations/test.txt`, split 50/50 per class (deterministic).

3. **Create flat test set for Android app**

   ```bash
   python -m scripts.create_test_app
   ```

   This creates:
   - `data/test_app/images/` – **all test images in a single folder** (no class subdirs).
   - `data/test_app/test.txt` – annotation file (no header) with lines:

   ```text
   image_name class_id species breed_id
   ```

   There are **1837** entries matching the prepared test split.  
   You can optionally compress it for transfer to a phone:

   ```powershell
   Compress-Archive -Path '.\data\test_app' -DestinationPath '.\data\test_app.zip' -Force
   ```


## Models

Pre‑exported ONNX models live under `model/`:

- `model/model.onnx` – default model.
- `model/t8/fastvit.onnx` – FastViT model for TensorFlow Lite / Android (T8 variant).
- `model/performer/fastvit.onnx` – FastViT Performer model.

You can re‑export from the training code using:

```bash
python -m fastvit.export_model
python -m scripts.export_models
```


## Android app (Petclassify)

The Android demo app lives in the `Petclassify/` directory.

- Open `Petclassify` in Android Studio.
- Ensure the model assets (ONNX / TFLite or split `.data`) are correctly referenced in the app module.
- Deploy to a device or emulator.
- The app will:
  - Load the FastViT model.
  - Run classification for images from `test_app/images` (or camera/gallery).
  - Log evaluation lines via Logcat.

Log format (per image):

```text
2025-12-17 21:08:55.715  6434-6452  EVAL_RESULT  com.example.petclassify  D  0,48.3934,0,5,27,33,24
```

which encodes:

```text
true_label,inference_time_ms,pred_1,pred_2,pred_3,pred_4,pred_5
```

Collect these logs into a text file, e.g. `results/t8-emulator.txt`.


## Evaluating logs

Use `scripts/analyze_logs.py` to parse Android Logcat outputs and compute accuracy / latency metrics.

### Basic usage

From repo root (with venv activated):

```bash
python -m scripts.analyze_logs results/t8-emulator.txt
```

The script:
- Extracts all `EVAL_RESULT` lines.
- Parses true label, inference time, and top‑5 predictions.
- Computes:
  - Total images
  - Average inference time
  - **Micro** top‑1 and top‑5 accuracy
  - **Macro** top‑1 and top‑5 accuracy


## License
MIT

