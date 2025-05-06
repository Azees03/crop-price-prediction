import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError

def load_and_preprocess_data():
    """
    Load and preprocess the weather and crop price datasets.
    """
    # Load datasets
    weather_data = pd.read_csv('formatted_weather.csv')
    crop_prices = {
        'brinjal': pd.read_csv('brinjalprice.csv'),
        'potato': pd.read_csv('potatoprice.csv'),
        'apple': pd.read_csv('apple.csv'),
        'onion': pd.read_csv('onionprice.csv'),
        'tomato': pd.read_csv('tomatoprice.csv'),
        'coconut': pd.read_csv('coconut.csv'),
        'groundnut': pd.read_csv('groundnut.csv'),
        'maize': pd.read_csv('maize.csv'),
        'paddy': pd.read_csv('paddy.csv'),
        'turmeric': pd.read_csv('turmeric.csv'),
        'lemon': pd.read_csv('lemon.csv'),
        'orange': pd.read_csv('orange.csv'),
        'banagreen': pd.read_csv('bananagreen.csv'),
        'beans': pd.read_csv('beans.csv'),
        'beetroot': pd.read_csv('beetroot.csv'),
        'bittergourd': pd.read_csv('bittergourd.csv'),
        'cabbage': pd.read_csv('cabbage.csv'),
        'carrot': pd.read_csv('carrot.csv'),
        'ladiesfinger': pd.read_csv('ladies_finger.csv'),
        'mango': pd.read_csv('mango.csv'),
        'raddish': pd.read_csv('raddish.csv'),
        'tapioca': pd.read_csv('tapioca.csv')
    }
   
    # Preprocess dates
    weather_data['Price Date'] = pd.to_datetime(weather_data['Price Date'], format='%d-%b-%y', dayfirst=True)
    for crop_name, crop_df in crop_prices.items():
        crop_prices[crop_name]['Price Date'] = pd.to_datetime(crop_df['Price Date'], format='%d-%b-%y', dayfirst=True)
    
    return weather_data, crop_prices

def hybrid_arima_lstm_model(data, crop_name, n_timesteps=30, forecast_horizon=7):  # Changed to 7 days
    try:
        prices = data['Modal Price (Rs./Quintal)']
        
        # ARIMA Model
        arima_model = ARIMA(prices, order=(5,1,2))
        arima_results = arima_model.fit()
        
        # LSTM Prep
        arima_residuals = prices - arima_results.fittedvalues
        scaler = MinMaxScaler()
        scaled_residuals = scaler.fit_transform(arima_residuals.dropna().values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(n_timesteps, len(scaled_residuals)):
            X.append(scaled_residuals[i-n_timesteps:i])
            y.append(scaled_residuals[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # LSTM Model
        lstm_model = Sequential([
            LSTM(50, activation='relu', input_shape=(n_timesteps, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss=MeanSquaredError())
        
        # Train
        lstm_model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[EarlyStopping(patience=10)]
        )
        
        # Forecasting
        arima_forecast = arima_results.forecast(steps=forecast_horizon)
        
        last_sequence = scaled_residuals[-n_timesteps:].reshape(1, n_timesteps, 1)
        lstm_residuals = []
        for _ in range(forecast_horizon):
            next_residual = lstm_model.predict(last_sequence, verbose=0)
            lstm_residuals.append(next_residual[0,0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1] = next_residual[0,0]
        
        lstm_residuals = scaler.inverse_transform(np.array(lstm_residuals).reshape(-1, 1))
        combined_forecast = arima_forecast + lstm_residuals.flatten()
        
        # Save models
        os.makedirs("models", exist_ok=True)
        with open(f"models/arima_{crop_name}.pkl", "wb") as f:
            pickle.dump(arima_results, f)
        lstm_model.save(f"models/lstm_{crop_name}.h5")
        with open(f"models/scaler_{crop_name}.pkl", "wb") as f:
            pickle.dump(scaler, f)
            
        return {
            'forecast': [round(float(x), 2) for x in combined_forecast],
            'metrics': {
                'mse': mean_squared_error(y, lstm_model.predict(X, verbose=0)),
                'mae': mean_absolute_error(y, lstm_model.predict(X, verbose=0))
            }
        }
        
    except Exception as e:
        print(f"Error in {crop_name}: {str(e)}")
        return None

def main():
    # Load data
    weather_data, crop_prices = load_and_preprocess_data()
    
    # Results storage
    results = {}
    
    # Process each crop
    for crop_name, price_data in crop_prices.items():
        print(f"\nProcessing {crop_name}...")
        
        try:
            model_results = hybrid_arima_lstm_model(price_data, crop_name)
            
            if model_results:
                results[crop_name] = model_results
        
        except Exception as e:
            print(f"Error in processing {crop_name}: {e}")
            continue
    
    return results

if __name__ == "__main__":
    results = main()