import tensorflow as tf
import numpy as np
import pandas as pd
import joblib   # if you saved the scaler

# load model
model = tf.keras.models.load_model("rnn_lstm_letter_classifier.keras")

# load scaler

scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# load the file

df = pd.read_csv("/content/a.csv")

# keep only features you trained with
FEATURES = ["flex_1", "flex_2", "flex_3", "flex_4", "flex_5",
            "GYRx", "GYRy", "GYRz",
            "ACCx", "ACCy", "ACCz"]

df = df[FEATURES].dropna()

WINDOW, STEP = 50, 20

X_list = []
arr = df.values
for start in range(0, max(0, len(arr) - WINDOW + 1), STEP):
    end = start + WINDOW
    if end <= len(arr):
        X_list.append(arr[start:end])

X = np.stack(X_list)

X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)


pred_probs = model.predict(X)
pred_labels = pred_probs.argmax(axis=1)

decoded = le.inverse_transform(pred_labels)

print("Predicted letters:", decoded)
