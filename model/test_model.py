import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =====================================================
# 1️⃣ LOAD TRAINED MODELS
# =====================================================
print("🔹 Loading trained models and scaler...")
rf = joblib.load("model/artifacts/rf_model.pkl")
iso = joblib.load("model/artifacts/iso_model.pkl")
scaler = joblib.load("model/artifacts/scaler.pkl")
thresholds = joblib.load("model/artifacts/thresholds.pkl")

# =====================================================
# 2️⃣ LOAD OR CREATE TEST DATA
# =====================================================
train_data_path = "model/health_vitals.csv"
test_data_path = "model/test_data.csv"

if not os.path.exists(test_data_path):
    print("⚠️ No test_data.csv found — splitting from main dataset (10%)...")
    df = pd.read_csv(train_data_path)
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    df_test.to_csv(test_data_path, index=False)
    print(f"✅ New test dataset created with {len(df_test)} samples → {test_data_path}")
else:
    print("✅ Found existing test dataset.")
    df_test = pd.read_csv(test_data_path)

print(f"📊 Test samples: {len(df_test)}, Features: {len(df_test.columns)}")

# =====================================================
# 3️⃣ PREPROCESS DATA
# =====================================================
if "Timestamp" in df_test.columns:
    df_test = df_test.drop(columns=["Timestamp"])

# Extract target variable if available
y_true = None
if "Risk Category" in df_test.columns:
    y_true = df_test["Risk Category"]
    X = df_test.drop(columns=["Risk Category"])
else:
    X = df_test.copy()

# Encode Gender if present
if "Gender" in X.columns:
    X["Gender"] = X["Gender"].map({"Male": 1, "Female": 0})

# Select numeric columns only
X = X.select_dtypes(include=["int64", "float64"])

# =====================================================
# 4️⃣ SCALE DATA & PREDICT
# =====================================================
X_scaled = scaler.transform(X)

# Predict risk using trained model
y_pred = rf.predict(X_scaled)

# Predict anomalies using Isolation Forest
anomaly_flags = iso.predict(X_scaled)

# =====================================================
# 5️⃣ EVALUATION (if labels available)
# =====================================================
if y_true is not None:
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Model Accuracy on Test Data: {acc:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
else:
    print("\n⚠️ No 'Risk Category' in test data — skipping evaluation metrics.")

# =====================================================
# 6️⃣ PATIENT ALERTS
# =====================================================
print("\n🔔 Generating test alerts...")

alerts = []
for idx, row in X.iterrows():
    pid = row.get("Patient ID", f"Unknown_{idx}")
    risk = y_pred[list(X.index).index(idx)]
    anomaly = anomaly_flags[list(X.index).index(idx)]

    if anomaly == -1 and risk == "High Risk":
        msg = f"🔴 Patient {pid}: High Risk detected, but readings show anomaly — please recheck sensors!"
    elif anomaly == -1 and risk == "Low Risk":
        msg = f"🟡 Patient {pid}: Low Risk but anomalous readings detected — verify equipment."
    elif anomaly == 1 and risk == "High Risk":
        msg = f"🟠 Patient {pid}: Clinically High Risk but sensor readings stable — monitor closely."
    else:
        msg = f"🟢 Patient {pid}: Normal readings. No issues detected."
    alerts.append(msg)

# Save alerts to CSV
os.makedirs("model/artifacts", exist_ok=True)
alerts_df = pd.DataFrame(alerts, columns=["Alert"])
alerts_df.to_csv("model/artifacts/test_alerts_output.csv", index=False)

print("✅ Alerts saved to model/artifacts/test_alerts_output.csv")

print("\n🔍 Sample Alerts:")
print("\n".join(alerts[:10]))
