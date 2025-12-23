import pandas as pd
import numpy as np
import os
import sys
import argparse
import re

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def analyze_pet_evaluation_logs(log_file):
    """
    Parses the evaluation logs from the Android app and calculates key
    performance metrics for the pet classification model.
    
    The script expects log lines in one of the following formats:
    - EVAL_RESULT: true_label,inference_time_ms,pred_1,pred_2,pred_3,pred_4,pred_5
    - EVAL_RESULT ... D  true_label,inference_time_ms,pred_1,pred_2,pred_3,pred_4,pred_5
    """
    if not os.path.exists(log_file):
        print(f"❌ Error: Log file not found at '{log_file}'")
        print("Please make sure you have saved your Logcat output to this file.")
        return

    records = []
    print(f"📄 Reading log file: {log_file}...")
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Find the start of our specific log message to ignore other noise
            if "EVAL_RESULT" in line:
                try:
                    # Handle two formats:
                    # 1. "EVAL_RESULT: 0,48.3934,0,5,27,33,24" (with colon)
                    # 2. "EVAL_RESULT ... D  0,48.3934,0,5,27,33,24" (without colon, Android logcat format)
                    if "EVAL_RESULT:" in line:
                        # Format 1: Has colon
                        data_part = line.split("EVAL_RESULT:")[1].strip()
                    else:
                        # Format 2: Extract data after "D" (which comes after EVAL_RESULT)
                        # Example: "2025-12-22 20:05:31.112 15122-15199 EVAL_RESULT ... D  0,47.432969,0,5,27,33,24"
                        # Find "EVAL_RESULT" first, then look for " D " after it
                        eval_index = line.find("EVAL_RESULT")
                        if eval_index != -1:
                            # Look for " D " after EVAL_RESULT
                            line_after_eval = line[eval_index:]
                            d_index = line_after_eval.find(" D ")
                            if d_index == -1:
                                d_index = line_after_eval.find(" D\t")
                            if d_index != -1:
                                # Extract everything after " D " or " D\t"
                                data_part = line_after_eval[d_index + 3:].strip()
                            else:
                                # Fallback: try to find comma-separated data after EVAL_RESULT
                                # Look for the first occurrence of a pattern like "digit,digit" or "digit,float"
                                match = re.search(r'(\d+,\d+[.,]\d+|\d+,\d+)', line_after_eval)
                                if match:
                                    data_part = line_after_eval[match.start():].strip()
                                else:
                                    continue
                        else:
                            continue
                    
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
                    print(f"⚠️ Warning: Could not parse line, skipping. Error: {e}\n   Line: '{line.strip()}'")

    if not records:
        print("Error: No valid 'EVAL_RESULT' lines found in the log file.")
        return

    df = pd.DataFrame(records)
    print(f"Found {len(df)} valid records to analyze.")

    # --- Calculate Metrics ---
    total_images = len(df)
    avg_inference_time = df['inference_time'].mean()

    # Top-1 Accuracy: Check if the true label matches the first prediction
    df['is_top_1_correct'] = df['true_label'] == df['top_1_pred']
    top_1_accuracy = (df['is_top_1_correct'].sum() / total_images) * 100

    # Top-5 Accuracy: Check if the true label is within the list of top 5 predictions
    df['is_top_5_correct'] = df.apply(lambda row: row['true_label'] in row['top_5_preds'], axis=1)
    top_5_accuracy = (df['is_top_5_correct'].sum() / total_images) * 100

    # --- Print Final Report ---
    print("\n========================================")
    print("Offline Evaluation Complete")
    print("========================================")
    print(f"📊 Total Images:      {total_images}")
    print(f"⏱️ Avg Inference Time: {avg_inference_time:.2f} ms")
    print(f"🎯 Top-1 Accuracy:     {top_1_accuracy:.2f}%")
    print(f"🎯 Top-5 Accuracy:     {top_5_accuracy:.2f}%")
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze pet classification evaluation logs')
    parser.add_argument('log_file', nargs='?', default='evaluation_logs.txt',
                        help='Path to the log file (default: evaluation_logs.txt)')
    args = parser.parse_args()
    
    analyze_pet_evaluation_logs(args.log_file)