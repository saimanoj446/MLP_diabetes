import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        pass 

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = np.sign(linear_output)
        return np.where(y_predicted == -1, 0, 1)

app = Flask(__name__)
CORS(app)

try:
    with open('naive_bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    
    with open('mlp_model.pkl', 'rb') as f:
        mlp_model = pickle.load(f)
        
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("All models and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_type = data.get('model_type', 'naive_bayes')
    
    try:
        features = [
            float(data['glucose']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['age'])
        ]
    except (ValueError, KeyError):
        return jsonify({'error': 'Invalid input data'}), 400

    final_features = np.array([features])
    scaled_features = scaler.transform(final_features)

    if model_type == 'naive_bayes':
        prediction = nb_model.predict(scaled_features)
    elif model_type == 'perceptron':
        prediction = mlp_model.predict(scaled_features)
    else:
        return jsonify({'error': 'Invalid model type'}), 400

    result_text = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    return jsonify({'diabetes_type': result_text, 'raw_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)