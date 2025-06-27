import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load full dataset
df = pd.read_csv("results/labeled_attack_dataset.csv")

# Separate normal and anomaly samples
normal_df = df[df['label'] == 0]
anomaly_df = df[df['label'] == 1]

print(f"Original count → Normal: {len(normal_df)}, Anomalies: {len(anomaly_df)}")

# --- Control this ratio ---
# You can change `test_anomaly_ratio` to adjust balance
test_anomaly_ratio = 0.5  # 0.5 = 50% anomalies in test set

# Determine test size
test_size = 0.2  # 20% test split

# Split each class separately
normal_train, normal_test = train_test_split(normal_df, test_size=test_size, random_state=42)
anomaly_train, anomaly_test = train_test_split(anomaly_df, test_size=test_size, random_state=42)

# Optionally balance the test set (artificially for fair eval)
# Limit the number of normal samples to match desired anomaly ratio
n_anomaly = len(anomaly_test)
n_normal_target = int(n_anomaly / test_anomaly_ratio - n_anomaly)
normal_test_balanced = normal_test.sample(n=n_normal_target, random_state=42)

# Final test set
test_df = pd.concat([normal_test_balanced, anomaly_test]).sample(frac=1, random_state=42).reset_index(drop=True)

# Final training set (rest of the data)
train_df = pd.concat([normal_train, anomaly_train]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Final test set → Normal: {len(normal_test_balanced)}, Anomalies: {len(anomaly_test)}")
print(f"📊 Test anomaly ratio: {len(anomaly_test) / len(test_df):.2f}")

# === Save or use the results ===
# Separate features and labels
X_train = train_df.drop(columns=["label", "label_type"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label", "label_type"])
y_test = test_df["label"]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optional: Save to disk
pd.DataFrame(X_train_scaled, columns=X_train.columns).assign(label=y_train.values).to_csv("results/train_balanced.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).assign(label=y_test.values).to_csv("results/test_balanced.csv", index=False)

print("\n📁 Saved balanced train/test sets to `results/` folder.")
