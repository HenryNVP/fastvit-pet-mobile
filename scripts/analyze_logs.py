import pandas as pd
import numpy as np
import os
import sys
import re
from pathlib import Path

DEFAULT_LOG_FILE = "evaluation_logs.txt"


def analyze_pet_evaluation_logs(log_file: str | None = None):
    """
    Parses the evaluation logs from the Android app and calculates key
    performance metrics for the pet classification model.

    The script expects Android Logcat-style lines like:

        2025-12-17 21:08:55.715  6434-6452  EVAL_RESULT  com.example.petclassify  D  0,48.3934,0,5,27,33,24

    which encode:

        true_label,inference_time_ms,pred_1,pred_2,pred_3,pred_4,pred_5
    """
    log_path = Path(log_file) if log_file is not None else Path(DEFAULT_LOG_FILE)
    if not log_path.exists():
        print(f"[ERROR] Log file not found at '{log_path}'")
        print("Please make sure you have saved your Logcat output to this file.")
        return

    records = []
    print(f"[INFO] Reading log file: {log_path}...")
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Find the start of our specific log message to ignore other noise
            if "EVAL_RESULT" in line:
                try:
                    # Extract the numeric payload after the Android log fields.
                    # Example line:
                    #   2025-12-17 ... EVAL_RESULT com.example.petclassify  D  0,48.3934,0,5,27,33,24
                    # We want just:
                    #   0,48.3934,0,5,27,33,24
                    match = re.search(r"EVAL_RESULT.*?D\s+([0-9].*)", line)
                    if not match:
                        raise ValueError("Could not find payload after 'EVAL_RESULT'")

                    data_part = match.group(1).strip()
                    parts = [p.strip() for p in data_part.split(',')]

                    true_label = int(parts[0])
                    inference_time = float(parts[1])
                    # The rest of the parts are the top 5 predicted indices
                    predictions = [int(p) for p in parts[2:]]

                    records.append({
                        'true_label': true_label,
                        'inference_time': inference_time,
                        'top_1_pred': predictions[0],
                        'top_5_preds': predictions
                    })
                except (IndexError, ValueError) as e:
                    print(f"[WARN] Could not parse line, skipping. Error: {e}\n   Line: '{line.strip()}'")

    if not records:
        print("[ERROR] No valid 'EVAL_RESULT' lines found in the log file.")
        return

    df = pd.DataFrame(records)
    print(f"[INFO] Found {len(df)} valid records to analyze.")

    # --- Calculate Metrics ---
    total_images = len(df)
    avg_inference_time = df['inference_time'].mean()

    # Top-1 Accuracy (micro): over all samples
    df['is_top_1_correct'] = df['true_label'] == df['top_1_pred']
    micro_top_1 = (df['is_top_1_correct'].sum() / total_images) * 100

    # Top-5 Accuracy (micro): over all samples
    df['is_top_5_correct'] = df.apply(lambda row: row['true_label'] in row['top_5_preds'], axis=1)
    micro_top_5 = (df['is_top_5_correct'].sum() / total_images) * 100

    # --- Macro accuracies ---
    # Per-class top-1 accuracy, then simple average across classes
    per_class_top1 = df.groupby('true_label')['is_top_1_correct'].mean() * 100.0
    macro_top_1 = per_class_top1.mean()

    # Per-class top-5 accuracy, then simple average across classes
    per_class_top5 = df.groupby('true_label')['is_top_5_correct'].mean() * 100.0
    macro_top_5 = per_class_top5.mean()

    # --- Print Final Report ---
    print("\n========================================")
    print("Offline Evaluation Complete")
    print("========================================")
    print(f"Total Images:              {total_images}")
    print(f"Avg Inference Time:        {avg_inference_time:.2f} ms")
    print("----------------------------------------")
    print(f"Micro Top-1 Accuracy:      {micro_top_1:.2f}%")
    print(f"Micro Top-5 Accuracy:      {micro_top_5:.2f}%")
    print(f"Macro Top-1 Accuracy:      {macro_top_1:.2f}%")
    print(f"Macro Top-5 Accuracy:      {macro_top_5:.2f}%")
    print("========================================")


if __name__ == "__main__":
    # Optional CLI usage: python analyze_logs.py path/to/log.txt
    log_arg = sys.argv[1] if len(sys.argv) > 1 else None
    analyze_pet_evaluation_logs(log_arg)