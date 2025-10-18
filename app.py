from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder='web')

lr_model = joblib.load('models/linear_regression_model.joblib')
rf_model = joblib.load('models/random_forest_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        temp = float(data['Temperature'])
        hum = float(data['Humidity'])
        sun = float(data['Sunlight'])
        co2_car = float(data['CO2_Emitted_Cars'])
        co2_tree = float(data['CO2_Absorbed_Trees'])
        o2_tree = float(data['O2_Released_Trees'])
        features = np.array([[temp, hum, sun, co2_car, co2_tree, o2_tree]])

        lr_pred = lr_model.predict(features)[0]
        rf_pred = rf_model.predict(features)[0]
        final_score = round((lr_pred + rf_pred) / 2, 2)
        return jsonify({'Carbon_Credit_Score': final_score})
    except Exception as e:
        return jsonify({'error': str(e)})
# Temporary storage for live sensor data
sensor_data = {}

@app.route('/update_sensor', methods=['POST'])
def update_sensor():
    """
    Receives sensor readings from ESP32
    Example payload:
    {
        "Temperature": 28.5,
        "Humidity": 60,
        "Sunlight": 450,
        "CO2_Emitted_Cars": 300,
        "CO2_Absorbed_Trees": 200,
        "O2_Released_Trees": 400
    }
    """
    global sensor_data
    sensor_data = request.json
    return jsonify({"status": "OK", "received": sensor_data})

@app.route('/get_sensor', methods=['GET'])
def get_sensor():
    """
    Returns the latest sensor readings to the web page
    """
    return jsonify(sensor_data)


if __name__ == '__main__':
    app.run(debug=True)
