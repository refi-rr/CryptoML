# core/model_trainer.py - UPDATE dengan method train_models yang lengkap

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, lookback_days=60, forecast_days=7):
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.models = {}
        self.histories = {}
        self.scalers = {}
    
    def get_callbacks(self, patience=15, reduce_lr_patience=10):
        """Enhanced callbacks with early stopping and learning rate scheduling"""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=0.0001,
                verbose=1
            )
        ]
    
    def prepare_sequences(self, data, feature_columns):
        """Prepare sequences for time series prediction dengan validation"""
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
        """Enhanced LSTM model dengan regularization"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            LSTM(64, return_sequences=False,
                 kernel_regularizer=l2(0.001)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.forecast_days)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_bilstm_model(self, input_shape):
        """Bidirectional LSTM model untuk better context understanding"""
        model = Sequential([
            Bidirectional(
                LSTM(64, return_sequences=True, 
                     kernel_regularizer=l2(0.001),
                     recurrent_regularizer=l2(0.001)),
                input_shape=input_shape
            ),
            BatchNormalization(),
            Dropout(0.3),
            
            Bidirectional(
                LSTM(32, return_sequences=False,
                     kernel_regularizer=l2(0.001))
            ),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dense(self.forecast_days)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_cnn_bilstm_model(self, input_shape):
        """Hybrid CNN-BiLSTM untuk capture local + temporal patterns"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dense(self.forecast_days)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def build_cnn_lstm_model(self, input_shape):
        """Enhanced CNN-LSTM model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            BatchNormalization(),
            Dense(self.forecast_days)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    # ✅ TAMBAHKAN METHOD train_models YANG LENGKAP
    def train_models(self, X_train, y_train, X_test, y_test, feature_names, selected_models):
        """Enhanced model training dengan early stopping dan volatility adaptation"""
        # Reset histories
        self.histories = {}
        
        # Calculate data volatility untuk adaptive training
        if len(y_train) > 0:
            price_volatility = np.std(y_train) / np.mean(y_train)
            high_volatility = price_volatility > 0.1
        else:
            high_volatility = False
        
        # Adaptive parameters berdasarkan volatility
        if high_volatility:
            print("🔄 High volatility regime detected - Using conservative training")
            epochs = 100
            patience = 20
        else:
            print("🔵 Normal volatility regime - Using standard training")  
            epochs = 80
            patience = 15
        
        callbacks = self.get_callbacks(patience=patience, reduce_lr_patience=patience-5)
        
        try:
            # LSTM
            if 'LSTM' in selected_models and len(X_train) > 0:
                print("Training LSTM...")
                lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
                history = lstm_model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=0,
                    shuffle=False
                )
                self.models['lstm'] = lstm_model
                self.histories['lstm'] = history
            
            # BiLSTM  
            if 'BiLSTM' in selected_models and len(X_train) > 0:
                print("Training BiLSTM...")
                bilstm_model = self.build_bilstm_model((X_train.shape[1], X_train.shape[2]))
                history = bilstm_model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=0,
                    shuffle=False
                )
                self.models['bilstm'] = bilstm_model
                self.histories['bilstm'] = history
            
            # CNN-BiLSTM
            if 'CNN-BiLSTM' in selected_models and len(X_train) > 0:
                print("Training CNN-BiLSTM...")
                cnn_bilstm_model = self.build_cnn_bilstm_model((X_train.shape[1], X_train.shape[2]))
                history = cnn_bilstm_model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=0,
                    shuffle=False
                )
                self.models['cnn_bilstm'] = cnn_bilstm_model
                self.histories['cnn_bilstm'] = history
            
            # CNN-LSTM
            if 'CNN-LSTM' in selected_models and len(X_train) > 0:
                print("Training CNN-LSTM...")
                cnn_lstm_model = self.build_cnn_lstm_model((X_train.shape[1], X_train.shape[2]))
                history = cnn_lstm_model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=0,
                    shuffle=False
                )
                self.models['cnn_lstm'] = cnn_lstm_model
                self.histories['cnn_lstm'] = history
            
            # Traditional ML Models
            if any(model in selected_models for model in ['Random Forest', 'Linear Regression', 'SVR']):
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                
                if 'Random Forest' in selected_models:
                    print("Training Random Forest...")
                    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                    rf_model.fit(X_train_flat, y_train[:, 0])  # Predict first day
                    self.models['random_forest'] = rf_model
                
                if 'Linear Regression' in selected_models:
                    print("Training Linear Regression...")
                    lr_model = LinearRegression()
                    lr_model.fit(X_train_flat, y_train[:, 0])
                    self.models['linear_regression'] = lr_model
                
                if 'SVR' in selected_models:
                    print("Training SVR...")
                    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                    svr_model.fit(X_train_flat, y_train[:, 0])
                    self.models['svr'] = svr_model
                    
            print(f"✅ Training completed. {len(self.models)} models trained.")
            print(f"✅ Models: {list(self.models.keys())}")
            print(f"✅ Histories: {list(self.histories.keys())}")
                    
        except Exception as e:
            print(f"Error in model training: {e}")
            import traceback
            traceback.print_exc()
        
        return self.models

    # ✅ TAMBAHKAN METHOD evaluate_models
    def evaluate_models(self, X_test, y_test):
        """Enhanced model evaluation dengan comprehensive metrics"""
        metrics = {}
        
        for name, model in self.models.items():
            try:
                if name in ['lstm', 'bilstm', 'cnn_lstm', 'cnn_bilstm']:
                    y_pred = model.predict(X_test, verbose=0)
                else:
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)
                    y_pred_single = model.predict(X_test_flat)
                    y_pred = np.column_stack([y_pred_single] * min(self.forecast_days, y_test.shape[1]))
                
                # Calculate metrics for each forecast day
                day_metrics = {}
                for day in range(min(self.forecast_days, y_pred.shape[1])):
                    if len(y_test) > day:
                        mae = mean_absolute_error(y_test[:, day], y_pred[:, day])
                        rmse = np.sqrt(mean_squared_error(y_test[:, day], y_pred[:, day]))
                        r2 = r2_score(y_test[:, day], y_pred[:, day])
                        
                        # Additional metrics
                        mape = np.mean(np.abs((y_test[:, day] - y_pred[:, day]) / y_test[:, day])) * 100
                    else:
                        mae = rmse = r2 = mape = 0
                    
                    day_metrics[f'day_{day+1}'] = {
                        'mae': mae,
                        'rmse': rmse, 
                        'r2': r2,
                        'mape': mape
                    }
                
                # Add training history info jika available
                training_info = {}
                if name in self.histories:
                    history = self.histories[name]
                    training_info = {
                        'final_epochs': len(history.history['loss']),
                        'final_train_loss': history.history['loss'][-1],
                        'final_val_loss': history.history['val_loss'][-1],
                        'best_val_loss': min(history.history['val_loss'])
                    }
                
                metrics[name] = {
                    'day_metrics': day_metrics,
                    'training_info': training_info
                }
                
                print(f"✅ {name} evaluation completed")
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                metrics[name] = {
                    'day_metrics': {f'day_{i+1}': {'mae': 0, 'rmse': 0, 'r2': 0, 'mape': 0} 
                                  for i in range(self.forecast_days)},
                    'training_info': {}
                }
        
        return metrics

    # ✅ TAMBAHKAN METHOD plot_training_history
    def plot_training_history(self, model_name):
        """Plot training history untuk model diagnostics"""
        if model_name in self.histories:
            history = self.histories[model_name]
            
            try:
                import plotly.graph_objects as go
            except ImportError:
                print("Plotly not available for training history plots")
                return None
            
            fig = go.Figure()
            
            # Plot loss
            fig.add_trace(go.Scatter(
                y=history.history['loss'],
                name='Training Loss',
                line=dict(color='blue', width=2)
            ))
            
            # Plot validation loss jika ada
            if 'val_loss' in history.history:
                fig.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    name='Validation Loss',
                    line=dict(color='red', width=2)
                ))
            
            fig.update_layout(
                title=f'{model_name.upper()} Training History',
                xaxis_title='Epochs',
                yaxis_title='Loss',
                template='plotly_white',
                height=400
            )
            
            return fig
        return None

    # ✅ TAMBAHKAN METHOD predict_future_prices
    def predict_future_prices(self, data_with_features, feature_names, forecast_days):
        """Make actual predictions using trained models"""
        predictions = {}
        
        try:
            # Prepare the last sequence for prediction
            features = data_with_features[feature_names].values
            current_price = data_with_features['close'].iloc[-1]
            
            # Ensure we have enough data for lookback
            if len(features) < self.lookback_days:
                raise ValueError(f"Not enough data for prediction. Need {self.lookback_days} days, got {len(features)}")
            
            last_sequence = features[-self.lookback_days:]
            last_sequence = last_sequence.reshape(1, self.lookback_days, len(feature_names))
            
            for model_name, model in self.models.items():
                try:
                    if model_name in ['lstm', 'bilstm', 'cnn_lstm', 'cnn_bilstm']:
                        # Deep learning models
                        y_pred = model.predict(last_sequence, verbose=0)[0]
                    else:
                        # Traditional ML models
                        X_flat = last_sequence.reshape(1, -1)
                        y_pred_single = model.predict(X_flat)[0]
                        # Create forecast for all days (repeat for simplicity)
                        y_pred = np.full(forecast_days, y_pred_single)
                    
                    # Ensure we only return the requested forecast days
                    y_pred = y_pred[:forecast_days]
                    
                    # Convert to proper display format
                    predictions[model_name] = y_pred
                    
                    print(f"✅ {model_name} prediction generated: {y_pred[:3]}...")
                    
                except Exception as e:
                    print(f"❌ Error predicting with {model_name}: {e}")
                    # Fallback to simple prediction based on last price
                    last_price = data_with_features['close'].iloc[-1]
                    predictions[model_name] = np.full(forecast_days, last_price)
            
            print(f"✅ Successfully generated predictions for {len(predictions)} models")
            
        except Exception as e:
            print(f"❌ Error in predict_future_prices: {e}")
            # Fallback for all models
            last_price = data_with_features['close'].iloc[-1]
            for model_name in self.models.keys():
                predictions[model_name] = np.full(forecast_days, last_price)
        
        return predictions
