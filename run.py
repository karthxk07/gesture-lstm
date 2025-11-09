
import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import shutil  # for moving files

# ===================== CONFIG =====================
WATCH_DIR = "/home/karthik/Projects/gesture-lstm/content"        # Folder to watch
TARGET_FILE = "data.csv"      # Editable filename
PROCESSED_SUBDIR = "processed"  # Subfolder name
CHECK_INTERVAL = 2            # Seconds between checks

# ===================== MODEL & SCALER =====================
model = tf.keras.models.load_model("rnn_lstm_letter_classifier.keras")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

FEATURES = [
    "flex_1", "flex_2", "flex_3", "flex_4", "flex_5",
    "GYRx", "GYRy", "GYRz",
    "ACCx", "ACCy", "ACCz"
]

WINDOW, STEP = 120, 20


# ===================== PROCESSING FUNCTION =====================
def process_file(file_path):
    print(f"\n📄 Processing: {file_path}")
    df = pd.read_csv(file_path)
    df = df[FEATURES].dropna()

    arr = df.values
    X_list = []

    for start in range(0, max(0, len(arr) - WINDOW + 1), STEP):
        end = start + WINDOW
        if end <= len(arr):
            X_list.append(arr[start:end])

    if not X_list:
        print("⚠️ No valid data windows found. Skipping file.")
        return

    X = np.stack(X_list)
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    pred_probs = model.predict(X)
    pred_labels = pred_probs.argmax(axis=1)
    decoded = le.inverse_transform(pred_labels)

    print("✅ Predicted letters:", decoded)

    # ========== Move and Rename File ==========
    processed_dir = os.path.join(WATCH_DIR, PROCESSED_SUBDIR)
    os.makedirs(processed_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_filename = f"data-processed-{timestamp}.csv"
    new_path = os.path.join(processed_dir, new_filename)

    shutil.move(file_path, new_path)
    print(f"📦 File moved and renamed to: {new_path}\n")


# ===================== WATCH LOOP =====================
print(f"👀 Watching directory: {WATCH_DIR}")
print(f"Waiting for file: {TARGET_FILE}")

while True:
    target_path = os.path.join(WATCH_DIR, TARGET_FILE)

    if os.path.exists(target_path):
        try:
            process_file(target_path)
        except Exception as e:
            print(f"❌ Error while processing: {e}")
        time.sleep(1)
    else:
        time.sleep(CHECK_INTERVAL)
