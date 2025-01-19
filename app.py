from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extract the values from the incoming JSON data
    acceleration = data.get('acceleration')
    rotation = data.get('rotation')
    magnetic_field = data.get('magnetic_field')
    light = data.get('light')

    # Convert the extracted values into a numpy array (ensure they are floats or ints)
    features = np.array([acceleration, rotation, magnetic_field, light]).reshape(1, -1)

    # Make prediction using the model
    prediction = model.predict(features)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
