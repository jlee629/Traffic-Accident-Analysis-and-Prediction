# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:30:37 2023

Group 6:
Alejandro Akifarry
Jungyu Lee
Minyoung Seol
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # front end
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app) # front end

# Load the pickled model
with open('best_model_rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    

@app.route("/")
def index():
    name = "Group 6 Prediction model"
    return render_template("index.html", name=name)
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Convert the dictionary directly to a dataframe
    features_df = pd.DataFrame([data])
    
    # Get the prediction
    prediction = model.predict(features_df)
    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
