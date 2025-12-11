import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import joblib
import os

# ====================================================
# 1. LOAD DATA
# ====================================================
print("🔹 Loading dataset...")
DATA_PATH = "model/health_dataset_synthetic.csv"
df = pd.read_csv(DATA_PATH)

# Drop unwanted columns
df = df.drop(columns=["Timestamp"], errors="ignore")

# Ensure correct target column name
target_col = "RiskLevel"
if target_col not in df.columns:
    raise ValueError(f"❌ Target column '{target_col}' not found in dataset!")

# Encode target labels (Low Risk=0, High Risk=1)
df[target_col] = df[target_col].replace({"Low Risk": 0, "High Risk": 1})

y = df[target_col]
X = df.drop(columns=[target_col])
X = X.select_dtypes(include=["int64", "float64"])

print(f"✅ Data Loaded. Samples: {len(X)}, Features: {len(X.columns)}")

# ====================================================
# 2. DATA SPLIT & SCALING
# ====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"🧠 Train size: {len(X_train)}, Test size: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====================================================
# 3. ANOMALY DETECTION MODEL
# ====================================================
print("🔹 Training Isolation Forest for anomaly detection...")
iso = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
iso.fit(X_train_scaled)

# ====================================================
# 4. RANDOM FOREST MODEL (Optimized)
# ====================================================
print("🔹 Training Random Forest for patient risk classification (optimized)...")

# Smaller param grid for faster training
param_grid = {
    "n_estimators": [80, 120],
    "max_depth": [8, 12],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [3, 5],
}

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=3, scoring="f1", verbose=1, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
best_rf = grid.best_estimator_
print(f"✅ Best RF Params: {grid.best_params_}")

# ====================================================
# 5. CROSS-VALIDATION
# ====================================================
cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=3, scoring="accuracy")
print(f"🔍 Cross-validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ====================================================
# 6. EVALUATION
# ====================================================
y_pred = best_rf.predict(X_test_scaled)
y_prob = best_rf.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n🔹 Model Performance:")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC-AUC: {roc:.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

if acc >= 0.99:
    print("⚠️ WARNING: Accuracy is extremely high. Dataset might be too simple or imbalanced.\n")

# ====================================================
# 7. SAVE MODELS & THRESHOLDS
# ====================================================
os.makedirs("model/artifacts", exist_ok=True)
joblib.dump(best_rf, "model/artifacts/rf_model.pkl")
joblib.dump(iso, "model/artifacts/iso_model.pkl")
joblib.dump(scaler, "model/artifacts/scaler.pkl")

rf_threshold = 0.65
iso_threshold = np.percentile(iso.decision_function(X_train_scaled), 5)
joblib.dump({"rf": rf_threshold, "iso": iso_threshold}, "model/artifacts/thresholds.pkl")
print(f"✅ Thresholds Saved | RF: {rf_threshold:.3f}, ISO: {iso_threshold:.3f}")

# ====================================================
# 8. ALERT GENERATION
# ====================================================
print("\n🔔 Generating patient-specific alerts...")
X_test_copy = X_test.copy()
iso_scores = iso.decision_function(X_test_scaled)
anomaly_flags = iso.predict(X_test_scaled)

alerts = []
for i, (pid, risk, iso_flag, iso_score) in enumerate(
    zip(X_test_copy.get("Patient ID", range(len(X_test_copy))), y_pred, anomaly_flags, iso_scores)
):
    if iso_flag == -1 and risk == 1:
        msg = f"🔴 Patient {pid}: High Risk detected with anomalous readings — recheck sensors."
    elif iso_flag == -1 and risk == 0:
        msg = f"🟡 Patient {pid}: Low Risk but anomalous readings detected — verify equipment."
    elif iso_flag == 1 and risk == 1:
        msg = f"🟠 Patient {pid}: Clinically High Risk but sensor readings stable — monitor closely."
    else:
        msg = f"🟢 Patient {pid}: Normal readings. No issues detected."
    alerts.append(msg)

alerts_df = pd.DataFrame(alerts, columns=["Alert"])
alerts_path = "model/artifacts/alerts_output.csv"
alerts_df.to_csv(alerts_path, index=False)

print(f"✅ Alerts saved to {alerts_path}\n")
print("🔍 Sample Alerts:")
print("\n".join(alerts[:10]))
