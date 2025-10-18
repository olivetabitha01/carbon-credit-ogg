from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained ML model (already deployed)
model = pickle.load(open('model.pkl', 'rb'))

# Temporary in-memory storage for sensor values
sensor_data = {}

@app.route('/')
def index():
    # Render your HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Calculate carbon credit using ML model
    """
    data = request.json
    temperature = float(data['Temperature'])
    humidity = float(data['Humidity'])
    sunlight = float(data['Sunlight'])
    co2_emitted = float(data['CO2_Emitted_Cars'])
    co2_absorbed = float(data['CO2_Absorbed_Trees'])
    o2_released = float(data['O2_Released_Trees'])

    # Arrange data for ML model
    features = np.array([[temperature, humidity, sunlight, co2_emitted, co2_absorbed, o2_released]])
    prediction = model.predict(features)[0]

    return jsonify({'credit': int(prediction)})

# ✅ New API to receive sensor data from ESP32
@app.route('/update_sensor', methods=['POST'])
def update_sensor():
    global sensor_data
    sensor_data = request.json
    return jsonify({"status": "success", "data_received": sensor_data})

# ✅ New API to send sensor data to frontend
@app.route('/get_sensor', methods=['GET'])
def get_sensor():
    return jsonify(sensor_data)

if __name__ == '__main__':
    app.run(debug=True)
