from flask import Flask, render_template, request
import pandas as pd
from model import WaterCapacityModel
import torch
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    land_area = float(request.form['land_area'])
    soil_moisture = float(request.form['soil_moisture'])
    rainfall = float(request.form['rainfall'])
    
    # Load the trained model
    model = WaterCapacityModel()
    model.load_state_dict(torch.load('water_capacity_model.pth'))
    model.eval()

    # Prepare input data for prediction
    input_data = np.array([[soil_moisture, land_area, rainfall]], dtype=np.float32)
    input_tensor = torch.from_numpy(input_data)

    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor).item()

    return render_template('result.html', prediction=prediction, land_area=land_area, soil_moisture=soil_moisture, rainfall=rainfall)

if __name__ == '__main__':
    app.run(debug=True)
