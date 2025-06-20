# === Run all tests ===
import csv
import random
from datetime import datetime
import numpy as np
import joblib
import os

# === Load your trained model and scaler ===
model, scaler = joblib.load("test_isolation_forest_model.pkl")  # adjust path if needed

# === Feature extraction matching training ===
def extract_features(data, window_size=10):
    features = []
    for i in range(0, len(data), window_size):
        window = data[i:i+window_size]
        values = [v for t, v in window]
        count_ones = sum(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        max_val = np.max(values)
        min_val = np.min(values)
        features.append([count_ones, mean_val, std_val, max_val, min_val])
    return features

# === Load CSV file ===
def load_csv(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return [(row[0], int(row[1])) for row in reader]

# === Tamper data by injecting 1s or 0s randomly ===
def tamper_data(data, mode="inject_1", rate=0.2):
    tampered = []
    for t, v in data:
        if random.random() < rate:
            v = 1 if mode == "inject_1" else 0
        tampered.append((t, v))
    return tampered

# === Generate DoS data - constant value for given seconds ===
def generate_dos_data(seconds=30, value=0, sample_interval=0.1):
    samples = int(seconds / sample_interval)
    return [(datetime.now().isoformat(), value) for _ in range(samples)]

# === Evaluate model on a data sample ===
def evaluate_scenario(data, label):
    features = extract_features(data, window_size=10)
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)  # 1 = normal, -1 = anomaly
    status = "Anomaly" if any(pred == -1 for pred in predictions) else "Normal"
    print(f"{label}: Model says ➤ {status}")
    return (label, status)

# === Main evaluation run ===
if __name__ == "__main__":
    results = []

    # Load datasets - update paths as needed
    synthetic_data = load_csv("simulated_vibration_log_20250617_100750.csv")
    abnormal_data = load_csv("abnormal_vibration_log_20250617_100907.csv")

    # Scenario 1: Synthetic Normal run
    results.append(evaluate_scenario(synthetic_data, "Synthetic Normal run"))

    # Scenario 2: Abnormal run
    results.append(evaluate_scenario(abnormal_data, "Abnormal run"))

    # Scenario 3: Replay attack (replay synthetic normal data)
    results.append(evaluate_scenario(synthetic_data, "Replay attack"))

    # Scenario 4: Tampering (inject 1s) into synthetic data
    tampered = tamper_data(synthetic_data, mode="inject_1", rate=0.3)
    results.append(evaluate_scenario(tampered, "Tampering - Inject 1s"))

    # Scenario 5: DoS - constant zeros
    dos = generate_dos_data(seconds=30, value=0)
    results.append(evaluate_scenario(dos, "DoS - Constant 0s"))

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_table.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Test Case", "Model Output"])
        writer.writerows(results)

    print("Evaluation complete. Results saved to results/evaluation_table.csv")
