# core/model_trainer.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, lookback_days=60, forecast_days=7):
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.models = {}
        self.scalers = {}
    
    def prepare_sequences(self, data, feature_columns):
        """Prepare sequences for time series prediction"""
        if len(data) < self.lookback_days + self.forecast_days:
            raise ValueError(f"Not enough data. Need {self.lookback_days + self.forecast_days} days, got {len(data)}")
        
        X, y = [], []
        features = data[feature_columns].values
        target = data['close'].values
        
        for i in range(self.lookback_days, len(data) - self.forecast_days):
            X.append(features[i-self.lookback_days:i])
            y.append(target[i:i+self.forecast_days])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.forecast_days)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse', metrics=['mae'])
        return model
    
    def build_cnn_lstm_model(self, input_shape):
        """Build CNN-LSTM hybrid model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(100, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(self.forecast_days)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_models(self, X_train, y_train, X_test, y_test, feature_names, selected_models):
        """Train multiple models"""
        results = {}
        
        try:
            # Random Forest
            if 'Random Forest' in selected_models:
                print("Training Random Forest...")
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                # Reshape for tree-based models
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                rf_model.fit(X_train_flat, y_train[:, 0])  # Predict first day only
                self.models['random_forest'] = rf_model
            
            # Linear Regression
            if 'Linear Regression' in selected_models:
                print("Training Linear Regression...")
                lr_model = LinearRegression()
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                lr_model.fit(X_train_flat, y_train[:, 0])
                self.models['linear_regression'] = lr_model
            
            # Support Vector Regression
            if 'SVR' in selected_models:
                print("Training SVR...")
                svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                svr_model.fit(X_train_flat, y_train[:, 0])
                self.models['svr'] = svr_model
            
            # LSTM
            if 'LSTM' in selected_models and len(X_train) > 0:
                print("Training LSTM...")
                lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
                history = lstm_model.fit(
                    X_train, y_train, 
                    epochs=100, 
                    batch_size=32,
                    validation_data=(X_test, y_test), 
                    verbose=0,
                    shuffle=False
                )
                self.models['lstm'] = lstm_model
            
            # CNN-LSTM
            if 'CNN-LSTM' in selected_models and len(X_train) > 0:
                print("Training CNN-LSTM...")
                cnn_lstm_model = self.build_cnn_lstm_model((X_train.shape[1], X_train.shape[2]))
                history_cnn = cnn_lstm_model.fit(
                    X_train, y_train, 
                    epochs=100, 
                    batch_size=32,
                    validation_data=(X_test, y_test), 
                    verbose=0,
                    shuffle=False
                )
                self.models['cnn_lstm'] = cnn_lstm_model
                
        except Exception as e:
            print(f"Error training models: {e}")
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        metrics = {}
        
        for name, model in self.models.items():
            try:
                if name in ['lstm', 'cnn_lstm']:
                    y_pred = model.predict(X_test)
                else:
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)
                    y_pred_single = model.predict(X_test_flat)
                    y_pred = np.column_stack([y_pred_single] * self.forecast_days)
                
                # Calculate metrics for each forecast day
                day_metrics = {}
                for day in range(min(self.forecast_days, y_pred.shape[1])):
                    if len(y_test) > 0:
                        mae = mean_absolute_error(y_test[:, day], y_pred[:, day])
                        rmse = np.sqrt(mean_squared_error(y_test[:, day], y_pred[:, day]))
                        r2 = r2_score(y_test[:, day], y_pred[:, day])
                    else:
                        mae = rmse = r2 = 0
                    
                    day_metrics[f'day_{day+1}'] = {
                        'mae': mae,
                        'rmse': rmse, 
                        'r2': r2
                    }
                
                metrics[name] = day_metrics
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                metrics[name] = {f'day_{i+1}': {'mae': 0, 'rmse': 0, 'r2': 0} for i in range(self.forecast_days)}
        
        return metrics
    
    def predict_future(self, model_name, last_sequence):
        """Make future predictions"""
        if model_name in self.models:
            model = self.models[model_name]
            if model_name in ['lstm', 'cnn_lstm']:
                return model.predict(np.array([last_sequence]))[0]
            else:
                return model.predict(last_sequence.reshape(1, -1))[0]
        return None
