import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

# Lazy load TF
tf = None
keras = None
layers = None

def load_tf():
    global tf, keras, layers
    if tf is None:
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as e:
            print(f"TensorFlow Import Failed: {e}")
            raise e


try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
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
            score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = attention_weights * inputs
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector
            
    TF_AVAILABLE = True

except ImportError:
    print("Warning: TensorFlow not found. Stock Prediction will be disabled.")
    TF_AVAILABLE = False
    AttentionLayer = object # Dummy


class StockPredictor:
    """Stock price predictor using Bidirectional LSTM with Attention (or Mock if TF unavailable)"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def prepare_features(self, data):
        """Feature Engineering"""
        df = pd.DataFrame(data)
        # Normalize columns to lowercase (Handle yfinance 'Close' vs 'close')
        df.columns = df.columns.str.lower()
        
        df['return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Technical Indicators
        # RSI
        rsi = RSIIndicator(close=df["close"], window=14)
        df["rsi"] = rsi.rsi()
        
        # MACD
        macd = MACD(close=df["close"])
        df["macd"] = macd.macd()
        
        # EMAs
        df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
        df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()
        
        # Volatility
        df["volatility"] = df["return"].rolling(window=20).std()
        
        df = df.dropna()
        return df[['close', 'volume', 'rsi', 'macd', 'volatility']]
        
    def create_sequences(self, data):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0])  # Predict close price
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build Bidirectional LSTM model with Attention"""
        if not TF_AVAILABLE: return None
        
        model = keras.Sequential([
            layers.Bidirectional(layers.LSTM(128, return_sequences=True, activation='tanh'), input_shape=input_shape),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, activation='tanh')),
            layers.Dropout(0.2),
            AttentionLayer(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model"""
        # Prepare features
        df = self.prepare_features(data)
        scaled_data = self.scaler.fit_transform(df.values)
        
        if not TF_AVAILABLE:
            print("Running in MOCK mode (TensorFlow missing). Skipping real training.")
            return {"loss": 0.001, "val_loss": 0.002}
        
        X, y = self.create_sequences(scaled_data)
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
            callbacks=[early_stop], verbose=0
        )
        return history
    
    def predict(self, data, days=1):
        """Make predictions for future days"""
        df = self.prepare_features(data)
        
        if not TF_AVAILABLE:
            print("Running in MOCK mode. Generating simulated predictions.")
            last_price = df['close'].iloc[-1]
            # Simple random walk simulation
            predictions = []
            current_price = last_price
            for _ in range(days):
                change = np.random.normal(0.001, 0.015) # Small drift, 1.5% volatility
                current_price = current_price * (1 + change)
                predictions.append(current_price)
            return np.array(predictions)
            
        scaled_data = self.scaler.transform(df.values)
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            current_batch = current_sequence.reshape(1, self.sequence_length, -1)
            pred_scaled = self.model.predict(current_batch, verbose=0)[0, 0]
            
            new_row = current_sequence[-1].copy()
            new_row[0] = pred_scaled  # Update close price
            
            current_sequence = np.vstack([current_sequence[1:], new_row])
            
            dummy_row = np.zeros((1, df.shape[1]))
            dummy_row[0, 0] = pred_scaled
            actual_price = self.scaler.inverse_transform(dummy_row)[0, 0]
            
            predictions.append(actual_price)
        
        return np.array(predictions)
