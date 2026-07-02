from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import pandas as pd

app = Flask(__name__)

# เปิด CORS
CORS(app)

# โหลดโมเดล
model = load("model.pkl")

FEATURES = [
    "URLLength",
    "IsHTTPS",
    "HasTitle",
    "HasFavicon",
    "NoOfURLRedirect",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfExternalRef"
]

@app.route("/")
def home():
    return "Random Forest API is Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    row = [data.get(feature, 0) for feature in FEATURES]

    df = pd.DataFrame([row], columns=FEATURES)

    prediction = int(model.predict(df)[0])

    probability = model.predict_proba(df)[0]

    confidence = round(float(max(probability) * 100), 2)

    risk = confidence if prediction == 0 else 100 - confidence

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "risk": round(risk, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)