import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import pandas as pd
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='.', static_url_path='')
logging.basicConfig(level=logging.INFO)

def load_models(crop_name):
    try:
        arima_path = f"models/arima_{crop_name}.pkl"
        lstm_path = f"models/lstm_{crop_name}.h5"
        
        if not os.path.exists(arima_path) or not os.path.exists(lstm_path):
            logging.error(f"Model files not found: {arima_path} or {lstm_path}")
            return None, None
        
        with open(arima_path, "rb") as f:
            arima_model = pickle.load(f)
        
        lstm_model = load_model(
            lstm_path,
            custom_objects={'MeanSquaredError': MeanSquaredError()},
            compile=True
        )
        
        return arima_model, lstm_model
    except Exception as e:
        logging.error(f"Error loading models for {crop_name}: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('web.html')

@app.route('/get_csv_data')
def get_csv_data():
    crop_name = request.args.get('crop', '').lower()
    crop_mapping = {
        'potato': 'potatoprice.csv',
        'tomato': 'tomatoprice.csv',
        'onion': 'onionprice.csv',
        'brinjal': 'brinjalprice.csv',
        'apple': 'apple.csv',
        'cabbage': 'cabbage.csv',
        'orange': 'orange.csv',
        'paddy': 'paddy.csv',
        'beetroot': 'beetroot.csv',
        'coconut': 'coconut.csv',
        'cotton': 'cotton.csv',
        'groundnut': 'groundnut.csv',
        'maize': 'maize.csv',
        'redchilli': 'redchilli.csv',
        'turmeric': 'turmeric.csv',
        'sunflower': 'sunflower.csv',
        'lemon': 'lemon.csv',
        'banagreen': 'bananagreen.csv',
        'beans': 'beans.csv',
        'bittergourd': 'bittergourd.csv',
        'carrot': 'carrot.csv',
        'ladiesfinger': 'ladies_finger.csv',
        'mango': 'mango.csv',
        'raddish': 'raddish.csv',
        'tapioca': 'tapioca.csv'
    }
    
    try:
        if crop_name in crop_mapping:
            csv_file = crop_mapping[crop_name]
            if not os.path.exists(csv_file):
                return jsonify([])
            
            crop_data = pd.read_csv(csv_file)
            if 'Price Date' in crop_data.columns:
                crop_data['Price Date'] = pd.to_datetime(crop_data['Price Date'], format='%d-%b-%y', dayfirst=True)
                crop_data.sort_values('Price Date', ascending=False, inplace=True)
            
            return jsonify(crop_data.head(4).to_dict('records'))
        return jsonify([])
    except Exception as e:
        logging.error(f"Error processing CSV data: {e}")
        return jsonify([])

@app.route('/get_prediction')
def get_prediction():
    crop_name = request.args.get('crop', '').lower()
    crop_mapping = {
        'potato': 'potatoprice.csv',
        'tomato': 'tomatoprice.csv',
        'onion': 'onionprice.csv',
        'brinjal': 'brinjalprice.csv',
        'apple': 'apple.csv',
        'cabbage': 'cabbage.csv',
        'orange': 'orange.csv',
        'paddy': 'paddy.csv',
        'beetroot': 'beetroot.csv',
        'coconut': 'coconut.csv',
        'cotton': 'cotton.csv',
        'groundnut': 'groundnut.csv',
        'maize': 'maize.csv',
        'redchilli': 'redchilli.csv',
        'turmeric': 'turmeric.csv',
        'sunflower': 'sunflower.csv',
        'lemon': 'lemon.csv',
        'banagreen': 'bananagreen.csv',
        'beans': 'beans.csv',
        'bittergourd': 'bittergourd.csv',
        'carrot': 'carrot.csv',
        'ladiesfinger': 'ladies_finger.csv',
        'mango': 'mango.csv',
        'raddish': 'raddish.csv',
        'tapioca': 'tapioca.csv'
    }
    
    try:
        if crop_name not in crop_mapping:
            return jsonify({'error': 'Invalid crop'}), 400
        
        csv_file = crop_mapping[crop_name]
        if not os.path.exists(csv_file):
            return jsonify({'error': 'Data file missing'}), 404

        arima_model, lstm_model = load_models(crop_name)
        if not arima_model or not lstm_model:
            return jsonify({'error': 'Models not found'}), 404

        with open(f"models/scaler_{crop_name}.pkl", "rb") as f:
            scaler = pickle.load(f)

        data = pd.read_csv(csv_file)
        data['Price Date'] = pd.to_datetime(data['Price Date'], format='%d-%b-%y')
        prices = data['Modal Price (Rs./Quintal)'].dropna()

        last_date = data['Price Date'].iloc[-1]
        date_strings = [(last_date + timedelta(days=i)).strftime('%d-%b-%Y') for i in range(1, 8)]

        arima_forecast = arima_model.forecast(steps=7)
        residuals = prices - arima_model.fittedvalues
        scaled_residuals = scaler.transform(residuals.values.reshape(-1, 1))

        last_sequence = scaled_residuals[-30:].reshape(1, 30, 1)
        lstm_residuals = []
        for _ in range(7):
            next_residual = lstm_model.predict(last_sequence, verbose=0)
            lstm_residuals.append(next_residual[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1] = next_residual[0, 0]

        lstm_residuals = scaler.inverse_transform(np.array(lstm_residuals).reshape(-1, 1))
        combined_forecast = arima_forecast + lstm_residuals.flatten()

        return jsonify({
            'predictions': [round(float(x), 2) for x in combined_forecast],
            'dates': date_strings
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)