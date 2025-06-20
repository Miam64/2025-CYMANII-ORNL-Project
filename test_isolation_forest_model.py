import csv
import joblib
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("Current working directory:", os.getcwd())

def load_csv(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return [(row[0], int(row[1])) for row in reader]

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

def main():
    synthetic_data_path = "simulated_vibration_log_20250617_100750.csv"
    normal_data = load_csv(synthetic_data_path)

    # Feature extraction
    X = extract_features(normal_data, window_size=10)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model training
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    # Save model + scaler
    joblib.dump((model, scaler), "test_isolation_forest_model.pkl")
    print("Model and scaler saved as test_isolation_forest_model.pkl")

if __name__ == "__main__":
    main()
