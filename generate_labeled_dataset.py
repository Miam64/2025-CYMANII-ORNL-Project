import random
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import skew, kurtosis

# === Feature extraction with label and optional overlapping windows ===
def extract_features_with_labels(data, label, label_name, window_size=10, step=5):
    features = []
    labels = []
    label_names = []
    for i in range(0, len(data) - window_size + 1, step):  # overlapping windows by step
        window = data[i:i + window_size]
        values = [v for t, v in window]

        count_ones = sum(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        max_val = np.max(values)
        min_val = np.min(values)

        if std_val < 1e-6:
            skewness = 0
            kurt_val = 0
        else:
            skewness = skew(values)
            kurt_val = kurtosis(values)

        zero_run = sum(1 for j in range(1, len(values)) if values[j - 1] == 1 and values[j] == 0)

        features.append([count_ones, mean_val, std_val, max_val, min_val, skewness, kurt_val, zero_run])
        labels.append(label)
        label_names.append(label_name)
    return features, labels, label_names

# === Load CSV file ===
def load_csv(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return [(row[0], int(row[1])) for row in reader]

# === Tamper data by injecting 1s or 0s randomly ===
def tamper_data(data, mode="inject_1", rate=0.5):
    tampered = []
    for t, v in data:
        if random.random() < rate:
            v = 1 if mode == "inject_1" else 0
        tampered.append((t, v))
    return tampered

# === Generate DoS data (constant value over time) ===
def generate_dos_data(seconds=30, value=0, sample_interval=0.1):
    samples = int(seconds / sample_interval)
    return [(datetime.now().isoformat(), value) for _ in range(samples)]

if __name__ == "__main__":
    all_features = []
    all_labels = []
    all_label_names = []

    def process_scenario(data, label_name, label_value, window_size=10, step=5):
        features, labels, label_names = extract_features_with_labels(
            data, label=label_value, label_name=label_name, window_size=window_size, step=step
        )
        all_features.extend(features)
        all_labels.extend(labels)
        all_label_names.extend(label_names)
        print(f"[{label_name}] → added {len(features)} samples labeled as {'Normal' if label_value == 1 else 'Attack'}")

    print("📥 Loading data...")
    real_data = load_csv("vibration_log_2.csv")              # Normal data
    abnormal_data = load_csv("vibration_log_abnormal1.csv")  # Actual attack samples

    print("🚀 Generating labeled scenarios...")

    # Normal data (non-overlapping windows)
    process_scenario(real_data, "Normal", label_value=1, window_size=10, step=10)

    # Abnormal attack data
    process_scenario(abnormal_data, "Abnormal Attack", label_value=0, window_size=10, step=5)

    # Replay attack (tampering with normal data but label as attack)
    process_scenario(real_data, "Replay Attack", label_value=0, window_size=10, step=5)

    # Multiple tampering runs with higher rate and overlapping windows
    for i in range(3):  # repeat 3 times with different tampering
        tampered = tamper_data(real_data, mode="inject_1", rate=0.5)
        process_scenario(tampered, f"Tampering Inject 1s Run {i+1}", label_value=0, window_size=10, step=5)

    # Multiple DoS attack runs with different durations
    for dur in [15, 30, 45]:
        dos = generate_dos_data(seconds=dur, value=0)
        process_scenario(dos, f"DoS Attack Constant 0s {dur}s", label_value=0, window_size=10, step=5)

    # Save to CSV
    df = pd.DataFrame(all_features, columns=[
        "count_ones", "mean", "std", "max", "min", "skew", "kurtosis", "zero_run"
    ])
    df['label'] = all_labels
    df['label_type'] = all_label_names

    os.makedirs("results", exist_ok=True)
    output_path = "results/labeled_attack_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Labeled dataset saved to {output_path}")
