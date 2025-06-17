from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)

# Load mô hình và scaler
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl") 

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])

    # Áp dụng one-hot nếu cần
    if "gender" in input_df.columns:
        input_df = pd.get_dummies(input_df, columns=["gender"], drop_first=True)

    # Chuẩn hoá
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
