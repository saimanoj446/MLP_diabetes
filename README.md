# GlucoGuard: Diabetes Prediction Web App

A full-stack machine learning application that predicts the likelihood of diabetes in patients based on diagnostic measures. This project integrates a **Multilayer Perceptron (MLP)** neural network and a **Naive Bayes** classifier into a Flask web interface for real-time risk assessment.

## Features
* **Dual-Model Backend:** utilizes both an MLP Neural Network and Naive Bayes classifier to ensure robust predictions.
* **Real-Time Inference:** Instant results served via a RESTful Flask API.
* **Data Preprocessing:** Automated feature scaling using a serialized Standard Scaler to maintain consistency between training and production data.
* **Interactive UI:** Simple and responsive HTML/CSS frontend for easy user interaction.

## Tech Stack
* **Language:** Python 3
* **Backend Framework:** Flask
* **Machine Learning:** Scikit-Learn (MLPClassifier, GaussianNB), Pandas, NumPy
* **Frontend:** HTML5, CSS3
* **Serialization:** Pickle

## Project Structure
```text
├── app.py                  # Main Flask application (API & Routing)
├── 2023A3PS0368H.ipynb     # Jupyter Notebook used for Data Analysis & Model Training
├── diabetes.csv            # Pima Indians Diabetes Dataset
├── mlp_model.pkl           # Serialized MLP Neural Network Model
├── naive_bayes_model.pkl   # Serialized Naive Bayes Model
├── scaler.pkl              # Serialized Standard Scaler for normalization
├── templates/
│   ├── index.html          # Main input form
│   └── frontend.html       # (Alternative/Result view)
└── README.md
