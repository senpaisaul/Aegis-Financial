import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

class AttentionLayer(layers.Layer):
    """Custom Attention Layer"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Calculate attention scores
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

class StockPredictor:
    """Stock price predictor using Bidirectional LSTM with Attention"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def prepare_features(self, data):
        """Prepare features with technical indicators"""
        df = data.copy()
        
        # Technical indicators
        df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Select features
        features = ['Close', 'Volume', 'SMA_10', 'SMA_20', 'EMA_10', 
                   'RSI', 'MACD', 'MACD_Signal', 'Price_Change', 'Volume_Change']
        
        df = df[features].dropna()
        return df
    
    def create_sequences(self, data, target_col_idx=0):
        """Create sequences for training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, target_col_idx])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build Bidirectional LSTM model with Attention"""
        model = keras.Sequential([
            # First Bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(128, return_sequences=True, activation='tanh'),
                input_shape=input_shape
            ),
            layers.Dropout(0.2),
            
            # Second Bidirectional LSTM layer
            layers.Bidirectional(
                layers.LSTM(64, return_sequences=True, activation='tanh')
            ),
            layers.Dropout(0.2),
            
            # Attention layer
            AttentionLayer(),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model"""
        # Prepare features
        df = self.prepare_features(data)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(df.values)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        return history
    
    def predict(self, data, days=1):
        """Make predictions for future days"""
        # Prepare features
        df = self.prepare_features(data)
        
        # Scale data
        scaled_data = self.scaler.transform(df.values)
        
        # Get last sequence
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape for prediction
            current_batch = current_sequence.reshape(1, self.sequence_length, -1)
            
            # Predict next value
            pred_scaled = self.model.predict(current_batch, verbose=0)[0, 0]
            
            # Create new row with predicted value
            new_row = current_sequence[-1].copy()
            new_row[0] = pred_scaled  # Update close price
            
            # Update sequence
            current_sequence = np.vstack([current_sequence[1:], new_row])
            
            # Inverse transform to get actual price
            dummy_row = np.zeros((1, df.shape[1]))
            dummy_row[0, 0] = pred_scaled
            actual_price = self.scaler.inverse_transform(dummy_row)[0, 0]
            
            predictions.append(actual_price)
        
        return np.array(predictions)
