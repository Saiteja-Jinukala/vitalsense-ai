from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import joblib
import os
import uuid

app = Flask(__name__)

# Load trained models
rf_model = joblib.load("model/artifacts/rf_model.pkl")
iso_model = joblib.load("model/artifacts/iso_model.pkl")
scaler = joblib.load("model/artifacts/scaler.pkl")

# Columns required by trained model
MODEL_COLS = [
    "Patient ID","Age","Gender","Heart Rate","Respiratory Rate",
    "Body Temperature","Oxygen Saturation","Systolic Blood Pressure",
    "Diastolic Blood Pressure","Weight (kg)","Height (m)",
    "Derived_BMI","Derived_MAP","Derived_Pulse_Pressure","Derived_HRV"
]

# --------------------------
# HOME PAGE
# --------------------------
@app.route("/")
def index():
    return render_template("index.html")

# --------------------------
# MANUAL INPUT PREDICTION
# --------------------------
@app.route("/manual_predict", methods=["POST"])
def manual_predict():
    try:
        # Read form inputs
        patient_data = {col: 0 for col in MODEL_COLS}

        patient_data["Patient ID"] = request.form.get("patient_id", 0)
        patient_data["Age"] = float(request.form.get("age"))
        patient_data["Gender"] = float(request.form.get("gender"))
        patient_data["Heart Rate"] = float(request.form.get("heartrate"))
        patient_data["Respiratory Rate"] = float(request.form.get("resp"))
        patient_data["Body Temperature"] = float(request.form.get("bodytemp"))
        patient_data["Oxygen Saturation"] = float(request.form.get("oxygen"))
        patient_data["Systolic Blood Pressure"] = float(request.form.get("sbp"))
        patient_data["Diastolic Blood Pressure"] = float(request.form.get("dbp"))
        patient_data["Weight (kg)"] = float(request.form.get("weight"))
        patient_data["Height (m)"] = float(request.form.get("height"))

        # Derived values
        patient_data["Derived_BMI"] = patient_data["Weight (kg)"] / (patient_data["Height (m)"] ** 2)
        patient_data["Derived_MAP"] = (patient_data["Diastolic Blood Pressure"]*2 + patient_data["Systolic Blood Pressure"]) / 3
        patient_data["Derived_Pulse_Pressure"] = patient_data["Systolic Blood Pressure"] - patient_data["Diastolic Blood Pressure"]
        patient_data["Derived_HRV"] = 60 / (patient_data["Heart Rate"] + 1e-5)

        df = pd.DataFrame([patient_data])
        X_scaled = scaler.transform(df)

        risk_pred = rf_model.predict(X_scaled)[0]
        risk_prob = rf_model.predict_proba(X_scaled)[0][1]
        iso_flag = iso_model.predict(X_scaled)[0]

        output = {
            "Patient ID": patient_data["Patient ID"],
            "Predicted Risk": "High Risk" if risk_pred == 1 else "Low Risk",
            "Confidence": f"{risk_prob*100:.2f}%",
            "Anomaly": "Anomaly" if iso_flag == -1 else "Normal"
        }

        return render_template("index.html", manual_result=output)

    except Exception as e:
        return render_template("index.html", error=f"❌ Error: {str(e)}")

# --------------------------
# CSV UPLOAD PREDICTION
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return render_template("index.html", error="❌ No file uploaded.")

        file = request.files["file"]
        df = pd.read_csv(file)

        # Rename short columns
        rename_map = {
            "SystolicBP": "Systolic Blood Pressure",
            "DiastolicBP": "Diastolic Blood Pressure",
            "BodyTemp": "Body Temperature",
            "OxygenLevel": "Oxygen Saturation",
            "HeartRate": "Heart Rate",
            "RespirationRate": "Respiratory Rate",
            "Weight": "Weight (kg)",
            "Height": "Height (m)",
            "BMI": "Derived_BMI"
        }
        df.rename(columns=rename_map, inplace=True)

        # Gender encoding
        if "Gender" in df.columns and df["Gender"].dtype == object:
            df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0)

        # Derived fields
        if "Derived_BMI" not in df.columns:
            df["Derived_BMI"] = df["Weight (kg)"] / (df["Height (m)"] ** 2)

        df["Derived_MAP"] = (df["Diastolic Blood Pressure"]*2 + df["Systolic Blood Pressure"]) / 3
        df["Derived_Pulse_Pressure"] = df["Systolic Blood Pressure"] - df["Diastolic Blood Pressure"]
        df["Derived_HRV"] = 60 / (df["Heart Rate"] + 1e-5)

        for col in MODEL_COLS:
            if col not in df.columns:
                df[col] = 0

        df = df[MODEL_COLS]

        # Predictions
        X_scaled = scaler.transform(df)
        risk_pred = rf_model.predict(X_scaled)
        risk_prob = rf_model.predict_proba(X_scaled)[:, 1]
        iso_flag = iso_model.predict(X_scaled)

        result = pd.DataFrame({
            "Patient ID": df["Patient ID"],
            "Predicted Risk": ["High Risk" if r == 1 else "Low Risk" for r in risk_pred],
            "Confidence": [f"{p*100:.2f}%" for p in risk_prob],
            "Anomaly": ["Anomaly" if f == -1 else "Normal" for f in iso_flag]
        })

        # Summary values
        summary = {
            "total": len(result),
            "high": (result["Predicted Risk"] == "High Risk").sum(),
            "low": (result["Predicted Risk"] == "Low Risk").sum(),
            "anomaly": (result["Anomaly"] == "Anomaly").sum(),
            "normal": (result["Anomaly"] == "Normal").sum(),
        }

        # Save CSV for download
        output_file = f"pred_output_{uuid.uuid4()}.csv"
        result.to_csv(output_file, index=False)

        return render_template(
            "index.html",
            table=result.to_html(index=False),
            summary=summary,
            download_file=output_file
        )

    except Exception as e:
        return render_template("index.html", error=f"❌ Error: {str(e)}")

# --------------------------
# FILE DOWNLOAD
# --------------------------
@app.route("/download/<filename>")
def download(filename):
    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
