import csv
import joblib
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.stats import skew, kurtosis

def load_csv(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return [(row[0], int(row[1])) for row in reader]

# === Improved feature extraction ===
def extract_features(data, window_size=10, step_size=5):
    features = []
    for i in range(0, len(data) - window_size + 1, step_size):  # Sliding window
        window = data[i:i+window_size]
        values = [v for t, v in window]
        count_ones = sum(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        max_val = np.max(values)
        min_val = np.min(values)
        skewness = skew(values)
        kurt = kurtosis(values)
        zero_run = sum(1 for j in range(1, len(values)) if values[j-1] == 1 and values[j] == 0)
        features.append([count_ones, mean_val, std_val, max_val, min_val, skewness, kurt, zero_run])
    return np.array(features)

def main():
    print("Current working directory:", os.getcwd())

    # === Load and preprocess data ===
    synthetic_data_path = "simulated_vibration_log_20.csv"
    data = load_csv(synthetic_data_path)
    X = extract_features(data, window_size=10, step_size=5)

    # === Scale features ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === Optional: Evaluate Isolation Forest before saving ===
    # We'll pseudo-label with IsolationForest just for evaluation
    model = IsolationForest(n_estimators=200, contamination=0.005,max_samples='auto', random_state=42)
    model.fit(X_scaled)
    preds = model.predict(X_scaled)
    preds = np.where(preds == 1, 1, 0)  # 1 = normal, 0 = anomaly

    # Check balance
    anomaly_ratio = np.mean(preds == 0)
    print(f"Estimated anomaly ratio in training data: {anomaly_ratio:.2%}")

    # (Optional) Train-test evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, preds, test_size=0.2, random_state=42)
    model_eval = IsolationForest(n_estimators=200, contamination=0.005,random_state=42)
    model_eval.fit(X_train)
    y_pred = model_eval.predict(X_test)
    y_pred = np.where(y_pred == 1, 1, 0)
    print("Evaluation on held-out data:")
    print(classification_report(y_test, y_pred))

    # === Save model and scaler ===
    joblib.dump((model, scaler), "test_isolation_forest_model.pkl")
    print("Model and scaler saved to test_isolation_forest_model.pkl")

if __name__ == "__main__":
    main()
