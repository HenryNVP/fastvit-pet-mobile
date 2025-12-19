import pandas as pd
import numpy as np
import os

LOG_FILE = "evaluation_logs.txt"

def analyze_pet_evaluation_logs():
    """
    Parses the evaluation logs from the Android app and calculates key
    performance metrics for the pet classification model.    The script expects log lines in the following format:
    EVAL_RESULT: true_label,inference_time_ms,pred_1,pred_2,pred_3,pred_4,pred_5
    """
    if not os.path.exists(LOG_FILE):
        print(f"❌ Error: Log file not found at '{LOG_FILE}'")
        print("Please make sure you have saved your Logcat output to this file.")
        return

    records = []
    print(f"📄 Reading log file: {LOG_FILE}...")
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            # Find the start of our specific log message to ignore other noise
            if "EVAL_RESULT" in line:
                try:
                    # Clean up the line and extract the comma-separated data
                    # Example: "D  EVAL_RESULT: 0,48.3934,0,5,27,33,24" -> "0,48.3934,0,5,27,33,24"
                    data_part = line.split("EVAL_RESULT:")[1].strip()
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
        print("❌ Error: No valid 'EVAL_RESULT' lines found in the log file.")
        return

    df = pd.DataFrame(records)
    print(f"📈 Found {len(df)} valid records to analyze.")

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
    print("✅ Offline Evaluation Complete!")
    print("========================================")
    print(f"📊 Total Images:      {total_images}")
    print(f"⏱️ Avg Inference Time: {avg_inference_time:.2f} ms")
    print(f"🎯 Top-1 Accuracy:     {top_1_accuracy:.2f}%")
    print(f"🎯 Top-5 Accuracy:     {top_5_accuracy:.2f}%")
    print("========================================")

if __name__ == "__main__":
    analyze_pet_evaluation_logs()