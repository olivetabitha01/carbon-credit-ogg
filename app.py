from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from datetime import datetime

# Flask app (ensure 'web' folder contains index.html)
app = Flask(__name__, template_folder='web')

# ===========================
# ✅ Load ML Models
# ===========================
try:
    lr_model = joblib.load('models/linear_regression_model.joblib')
    print("✅ Linear Regression model loaded successfully.")
except Exception as e:
    lr_model = None
    print(f"⚠️ Failed to load linear_regression_model.joblib: {e}")

try:
    rf_model = joblib.load('models/random_forest_model.joblib')
    print("✅ Random Forest model loaded successfully.")
except Exception as e:
    rf_model = None
    print(f"⚠️ Failed to load random_forest_model.joblib: {e}")

# ===========================
# 🌡️ Temporary Sensor Storage
# ===========================
sensor_data = {}

# ===========================
# 🏠 Home Route
# ===========================
@app.route('/')
def home():
    return render_template('index.html')

# ===========================
# 📊 Prediction Route
# ===========================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives JSON with parameter values and returns carbon credit score.
    """
    data = request.get_json()

    try:
        temp = float(data.get('Temperature', 0))
        hum = float(data.get('Humidity', 0))
        sun = float(data.get('Sunlight', 0))
        co2_car = float(data.get('CO2_Emitted_Cars', 0))
        co2_tree = float(data.get('CO2_Absorbed_Trees', 0))
        o2_tree = float(data.get('O2_Released_Trees', 0))

        features = np.array([[temp, hum, sun, co2_car, co2_tree, o2_tree]])

        # Predictions from both models
        preds = []
        if lr_model:
            preds.append(lr_model.predict(features)[0])
        if rf_model:
            preds.append(rf_model.predict(features)[0])

        if not preds:
            return jsonify({'error': 'No ML model loaded on server.'}), 500

        final_score = round(sum(preds) / len(preds), 2)
        return jsonify({'Carbon_Credit_Score': final_score})

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

# ===========================
# 📥 ESP32 Sensor Update Route
# ===========================
@app.route('/update_sensor', methods=['POST'])
def update_sensor():
    global sensor_data
    sensor_data = request.json or {}
    sensor_data['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"status": "OK", "received": sensor_data})

# ===========================
# 📤 Send Sensor Data to Frontend
# ===========================
@app.route('/get_sensor', methods=['GET'])
def get_sensor():
    return jsonify(sensor_data)

# ===========================
# 🚀 Run Server
# ===========================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
