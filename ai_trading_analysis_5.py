"""
Advanced Cryptocurrency Trading System with Deep Reinforcement Learning
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crypto_trading')

# Constants
TRANSACTION_FEE = 0.001  # 0.1% transaction fee


#-------------------------------------------------------------------------
# Data Module
#-------------------------------------------------------------------------

class DataProcessor:
    """Handles data downloading, preprocessing, and feature engineering."""
    
    def __init__(
        self, 
        ticker: str = 'TSLA', 
        start_date: str = '2020-01-01', 
        end_date: str = '2024-03-15',
        seq_length: int = 60,
        random_state: int = 42
    ):
        """
        Initialize the data processor.
        
        Args:
            ticker: The ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            seq_length: Length of sequence for time series data
            random_state: Random seed for reproducibility
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.seq_length = seq_length
        self.random_seed = random_state
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        # Scalers
        self.feature_scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()
        
        # Data containers
        self.raw_data = None
        self.processed_data = None
        self.scaled_features = None
        self.scaled_prices = None
        self.X = None  # Feature sequences
        self.y = None  # Target prices
    
    def download_data(self) -> pd.DataFrame:
        """Download historical data for the specified ticker."""
        logger.info(f"Downloading data for {self.ticker} from {self.start_date} to {self.end_date}")
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        if len(data) < self.seq_length + 1:
            raise ValueError(f"Insufficient data. Need at least {self.seq_length + 1} data points.")
        
        self.raw_data = data
        return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        df = data.copy()
        
        # Moving averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean().replace(0, 0.001)  # Avoid division by zero
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        
        # Volatility
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility_21'] = df['Log_Return'].rolling(window=21).std()
        
        # Volume indicators
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_MA_5'] = df['Price_Change'].rolling(window=5).mean()
        
        # Fill NaN values
        df = df.ffill().fillna(0)
        return df
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and preprocess financial data with enhanced features.
        
        Returns:
            Tuple containing X (features) and y (targets)
        """
        if self.raw_data is None:
            self.download_data()
        
        # Add technical indicators
        self.processed_data = self.add_technical_indicators(self.raw_data)
        
        # Select features for the model
        feature_columns = [
            'Close', 'Volume', 'High', 'Low', 
            'MA_7', 'MA_21', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'Volatility_21',
            'Volume_Change', 'Price_Change'
        ]
        
        # Scale features
        features_df = self.processed_data[feature_columns]
        self.scaled_features = self.feature_scaler.fit_transform(features_df)
        
        # Scale price for prediction targets
        price_df = self.processed_data[['Close']]
        self.scaled_prices = self.price_scaler.fit_transform(price_df)
        
        # Create sequences for time series data
        self.X, self.y = self._create_sequences(self.scaled_features, self.scaled_prices)
        
        logger.info(f"Data prepared: {len(self.processed_data)} rows, {len(feature_columns)} features")
        logger.info(f"Sequence shape: {self.X.shape}, Target shape: {self.y.shape}")
        
        return self.X, self.y
    
    def _create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Feature data
            targets: Target values
        
        Returns:
            Tuple of sequences and targets
        """
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(targets[i + self.seq_length, 0])
        return np.array(X), np.array(y)
    
    def save_scalers(self, model_dir: str) -> None:
        """Save scalers to disk for later use."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save feature scaler
        feature_scaler_path = os.path.join(model_dir, 'feature_scaler.json')
        feature_scaler_data = {
            "min_": self.feature_scaler.min_.tolist(),
            "scale_": self.feature_scaler.scale_.tolist()
        }
        with open(feature_scaler_path, 'w') as f:
            json.dump(feature_scaler_data, f)
        
        # Save price scaler
        price_scaler_path = os.path.join(model_dir, 'price_scaler.json')
        price_scaler_data = {
            "min_": self.price_scaler.min_.tolist(),
            "scale_": self.price_scaler.scale_.tolist()
        }
        with open(price_scaler_path, 'w') as f:
            json.dump(price_scaler_data, f)
        
        logger.info(f"Scalers saved to {model_dir}")
    
    def load_scalers(self, model_dir: str) -> None:
        """Load scalers from disk."""
        # Load feature scaler
        feature_scaler_path = os.path.join(model_dir, 'feature_scaler.json')
        with open(feature_scaler_path, 'r') as f:
            feature_scaler_data = json.load(f)
        
        self.feature_scaler = MinMaxScaler()
        self.feature_scaler.min_ = np.array(feature_scaler_data["min_"])
        self.feature_scaler.scale_ = np.array(feature_scaler_data["scale_"])
        
        # Load price scaler
        price_scaler_path = os.path.join(model_dir, 'price_scaler.json')
        with open(price_scaler_path, 'r') as f:
            price_scaler_data = json.load(f)
        
        self.price_scaler = MinMaxScaler()
        self.price_scaler.min_ = np.array(price_scaler_data["min_"])
        self.price_scaler.scale_ = np.array(price_scaler_data["scale_"])
        
        logger.info(f"Scalers loaded from {model_dir}")


#-------------------------------------------------------------------------
# Model Module
#-------------------------------------------------------------------------

# Define the PositionalEncoding custom layer outside the class
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
    
    def get_angles(self, pos, i, d_model):
        # Calculate the angles for positional encoding
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    def build(self, input_shape):
        # Create positional encoding matrix
        pos_encoding = np.zeros((self.max_len, self.d_model))
        positions = np.expand_dims(np.arange(0, self.max_len), 1)
        i = np.expand_dims(np.arange(0, self.d_model), 0)
        
        angles = self.get_angles(positions, i, self.d_model)
        
        # Apply sin to even indices
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        # Apply cos to odd indices
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
        
        self.pos_encoding = tf.cast(tf.convert_to_tensor(pos_encoding), dtype=tf.float32)
        self.pos_encoding = self.pos_encoding[tf.newaxis, ...]
    
    def call(self, inputs):
        # Add positional encoding to inputs
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model
        })
        return config


class PricePredictor:
    """Handles price prediction models (CNN, LSTM, and Transformer)."""
    
    def __init__(
        self, 
        seq_length: int, 
        num_features: int,
        model_dir: str = 'saved_models'
    ):
        """
        Initialize the price predictor.
        
        Args:
            seq_length: Length of input sequences
            num_features: Number of features per sequence
            model_dir: Directory to save models
        """
        self.seq_length = seq_length
        self.num_features = num_features
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.cnn_model = None
        self.lstm_model = None
        self.transformer_model = None
    
    def build_cnn_model(self) -> keras.Model:
        """Build an enhanced CNN model for price prediction."""
        input_layer = keras.layers.Input(shape=(self.seq_length, self.num_features))
        
        # Multiple convolutional layers with different kernel sizes (multi-scale features)
        conv1 = keras.layers.Conv1D(64, 2, activation='relu', padding='same')(input_layer)
        bn1 = keras.layers.BatchNormalization()(conv1)
        
        conv2 = keras.layers.Conv1D(64, 3, activation='relu', padding='same')(input_layer)
        bn2 = keras.layers.BatchNormalization()(conv2)
        
        conv3 = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(input_layer)
        bn3 = keras.layers.BatchNormalization()(conv3)
        
        # Merge different convolutional outputs
        merged = keras.layers.Concatenate()([bn1, bn2, bn3])
        
        # Use a more compatible approach for attention
        # Instead of MultiHeadAttention, use a combination of Conv1D and GlobalMaxPooling1D
        # to capture important features across the sequence
        attn_conv1 = keras.layers.Conv1D(128, 3, padding='same', activation='relu')(merged)
        attn_pool1 = keras.layers.GlobalMaxPooling1D()(attn_conv1)
        attn_dense1 = keras.layers.Dense(merged.shape[-1], activation='sigmoid')(attn_pool1)
        
        # Reshape to broadcast across the sequence dimension
        attn_reshape = keras.layers.Reshape((1, merged.shape[-1]))(attn_dense1)
        # Broadcast attention weights across the sequence
        attn_broadcast = keras.layers.RepeatVector(self.seq_length)(attn_dense1)
        
        # Apply attention weights
        weighted_features = keras.layers.Multiply()([merged, attn_broadcast])
        
        # Add residual connection
        enhanced_features = keras.layers.Add()([merged, weighted_features])
        enhanced_features = keras.layers.LayerNormalization()(enhanced_features)
        
        # Global pooling
        pooling = keras.layers.GlobalAveragePooling1D()(enhanced_features)
        
        # Deep fully connected layers
        dense1 = keras.layers.Dense(128, activation='relu')(pooling)
        dropout1 = keras.layers.Dropout(0.3)(dense1)
        
        dense2 = keras.layers.Dense(64, activation='relu')(dropout1)
        dropout2 = keras.layers.Dropout(0.3)(dense2)
        
        # Skip connection
        merged_features = keras.layers.Concatenate()([pooling, dropout2])
        
        # Output layer
        output = keras.layers.Dense(1, activation='linear')(merged_features)
        
        # Create and compile model
        model = keras.Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss=keras.losses.Huber()
        )
        
        return model
    
    def build_lstm_model(self) -> keras.Model:
        """Build an enhanced LSTM model for time series forecasting."""
        input_layer = keras.layers.Input(shape=(self.seq_length, self.num_features))
        
        # Bidirectional LSTM layers
        lstm1 = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(input_layer)
        
        lstm2 = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(lstm1)
        
        # Concat LSTM outputs (residual connection)
        concat = keras.layers.Concatenate()([lstm1, lstm2])
        
        # Self-attention mechanism
        attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(concat, concat)
        attention = keras.layers.LayerNormalization()(attention + concat)
        
        # Global pooling
        pooling = keras.layers.GlobalAveragePooling1D()(attention)
        
        # Fully connected layers
        dense1 = keras.layers.Dense(64, activation='relu')(pooling)
        dropout1 = keras.layers.Dropout(0.3)(dense1)
        
        # Output layer
        output = keras.layers.Dense(1, activation='linear')(dropout1)
        
        # Create and compile model
        model = keras.Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss=keras.losses.Huber()
        )
        
        return model
    
    def build_transformer_model(self) -> keras.Model:
        """Build a Transformer model for time series forecasting."""
        input_layer = keras.layers.Input(shape=(self.seq_length, self.num_features))
        
        # Project input to embedding dimension for self-attention
        embed_dim = 128  # Embedding dimension
        
        # Apply a time-distributed dense layer to project inputs to embedding dimension
        x = keras.layers.Dense(embed_dim)(input_layer)
        
        # Add positional encoding
        x = PositionalEncoding(self.seq_length, embed_dim)(x)
        
        # Apply dropout for regularization
        x = keras.layers.Dropout(0.1)(x)
        
        # Multi-head self-attention blocks with residual connections and layer normalization
        for i in range(4):  # 4 transformer blocks
            # Self-attention block
            attention_output = keras.layers.MultiHeadAttention(
                num_heads=8, 
                key_dim=embed_dim // 8,
                dropout=0.1
            )(x, x)
            
            # Add & Norm (residual connection and layer normalization)
            attention_output = keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed-forward network
            ffn = keras.Sequential([
                keras.layers.Dense(embed_dim * 4, activation='gelu'),
                keras.layers.Dense(embed_dim),
                keras.layers.Dropout(0.1)
            ])
            
            ffn_output = ffn(attention_output)
            
            # Add & Norm (another residual connection and layer normalization)
            x = keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        
        # Global pooling to handle variable length sequences
        pooled = keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layers with dropout for regularization
        dense1 = keras.layers.Dense(64, activation='gelu')(pooled)
        dropout1 = keras.layers.Dropout(0.2)(dense1)
        
        dense2 = keras.layers.Dense(32, activation='gelu')(dropout1)
        dropout2 = keras.layers.Dropout(0.2)(dense2)
        
        # Output layer
        output = keras.layers.Dense(1, activation='linear')(dropout2)
        
        # Create and compile model
        model = keras.Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),  # Clip gradients for stability
            loss=keras.losses.Huber()  # Huber loss is more robust to outliers
        )
        
        return model
    
    def train_models(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> None:
        """
        Train CNN, LSTM, and Transformer models.
        
        Args:
            X: Input sequences
            y: Target values
            epochs: Number of training epochs
        """
        # Build models
        self.cnn_model = self.build_cnn_model()
        self.lstm_model = self.build_lstm_model()
        self.transformer_model = self.build_transformer_model()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
        
        # Train CNN model
        logger.info("Training CNN model...")
        cnn_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(self.model_dir / 'best_cnn_model.keras'),
            save_best_only=True,
            monitor='val_loss'
        )
        
        self.cnn_model.fit(
            X, y,
            epochs=epochs,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping, cnn_checkpoint],
            verbose=1
        )
        
        # Load best CNN model
        self.cnn_model = keras.models.load_model(str(self.model_dir / 'best_cnn_model.keras'))
        
        # Train LSTM model
        logger.info("Training LSTM model...")
        lstm_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(self.model_dir / 'best_lstm_model.keras'),
            save_best_only=True,
            monitor='val_loss'
        )
        
        self.lstm_model.fit(
            X, y,
            epochs=epochs,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping, lstm_checkpoint, reduce_lr],
            verbose=1
        )
        
        # Load best LSTM model
        self.lstm_model = keras.models.load_model(str(self.model_dir / 'best_lstm_model.keras'))
        
        # Train Transformer model
        logger.info("Training Transformer model...")
        transformer_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(self.model_dir / 'best_transformer_model.keras'),
            save_best_only=True,
            monitor='val_loss'
        )
        
        # Use a learning rate scheduler for transformer training
        lr_scheduler = keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.98 if epoch > 3 else lr
        )
        
        try:
            self.transformer_model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,  # Smaller batch size for transformer model
                validation_split=0.2,
                callbacks=[early_stopping, transformer_checkpoint, lr_scheduler],
                verbose=1
            )
            
            # Load best Transformer model
            self.transformer_model = keras.models.load_model(str(self.model_dir / 'best_transformer_model.keras'))
            logger.info("Transformer model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            logger.warning("Continuing without Transformer model")
        
        logger.info("Price prediction models trained successfully")
    
    def load_models(self) -> None:
        """Load pre-trained models from disk."""
        cnn_path = self.model_dir / 'best_cnn_model.keras'
        lstm_path = self.model_dir / 'best_lstm_model.keras'
        transformer_path = self.model_dir / 'best_transformer_model.keras'
        
        # Custom objects for loading models with custom layers
        custom_objects = {'PositionalEncoding': PositionalEncoding}
        
        # Load CNN model
        if cnn_path.exists():
            try:
                self.cnn_model = keras.models.load_model(str(cnn_path))
                logger.info(f"CNN model loaded from {cnn_path}")
            except Exception as e:
                logger.error(f"Error loading CNN model: {e}")
        else:
            logger.warning(f"CNN model not found at {cnn_path}")
        
        # Load LSTM model
        if lstm_path.exists():
            try:
                self.lstm_model = keras.models.load_model(str(lstm_path))
                logger.info(f"LSTM model loaded from {lstm_path}")
            except Exception as e:
                logger.error(f"Error loading LSTM model: {e}")
        else:
            logger.warning(f"LSTM model not found at {lstm_path}")
        
        # Load Transformer model
        if transformer_path.exists():
            try:
                self.transformer_model = keras.models.load_model(
                    str(transformer_path),
                    custom_objects=custom_objects
                )
                logger.info(f"Transformer model loaded from {transformer_path}")
            except Exception as e:
                logger.error(f"Error loading Transformer model: {e}")
                logger.info("Attempting to build a new Transformer model...")
                try:
                    # Build a new model if loading fails
                    self.transformer_model = self.build_transformer_model()
                    logger.info("New Transformer model built successfully")
                except Exception as e2:
                    logger.error(f"Could not build Transformer model: {e2}")
        else:
            logger.warning(f"Transformer model not found at {transformer_path}")
    
    def predict_price(self, input_sequence: np.ndarray) -> float:
        """
        Predict future price using ensemble of CNN, LSTM, and Transformer models.
        
        Args:
            input_sequence: Input sequence of shape (seq_length, num_features)
                           or (1, seq_length, num_features)
        
        Returns:
            Predicted price
        """
        if input_sequence.ndim == 2:
            input_sequence = input_sequence[np.newaxis, :, :]
        
        if self.cnn_model is None or self.lstm_model is None:
            raise ValueError("Models not trained or loaded. Call train_models() or load_models() first.")
        
        try:
            # Get predictions from CNN and LSTM models
            pred_cnn = self.cnn_model.predict(input_sequence, verbose=0)[0, 0]
            pred_lstm = self.lstm_model.predict(input_sequence, verbose=0)[0, 0]
            
            # Initialize transformer prediction
            pred_transformer = None
            
            # Try to get prediction from transformer model if available
            if self.transformer_model is not None:
                try:
                    pred_transformer = self.transformer_model.predict(input_sequence, verbose=0)[0, 0]
                    logger.debug(f"Transformer prediction: {pred_transformer}")
                except Exception as e:
                    logger.warning(f"Error in transformer prediction: {e}")
                    pred_transformer = None
            
            # Ensemble prediction with dynamic weighting
            if pred_transformer is not None:
                # Weight models based on their predictive power (can be refined based on validation)
                weights = np.array([0.3, 0.3, 0.4])  # CNN, LSTM, Transformer
                predictions = np.array([pred_cnn, pred_lstm, pred_transformer])
                combined_pred = np.sum(weights * predictions)
                
                logger.debug(f"Ensemble prediction with Transformer: CNN={pred_cnn:.4f}, "
                           f"LSTM={pred_lstm:.4f}, Transformer={pred_transformer:.4f}, "
                           f"Combined={combined_pred:.4f}")
            else:
                # Use original ensemble if transformer is not available
                combined_pred = (pred_cnn + pred_lstm) / 2
                logger.debug(f"Ensemble prediction without Transformer: CNN={pred_cnn:.4f}, "
                           f"LSTM={pred_lstm:.4f}, Combined={combined_pred:.4f}")
            
            return combined_pred
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return None


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for more efficient learning."""
    
    def __init__(
        self, 
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 0.01
    ):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
            alpha: Degree of prioritization (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction factor
            beta_increment: Increment for beta parameter over time
            epsilon: Small value to avoid zero priority
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(
        self, 
        state: np.ndarray,
        action: Union[int, float],
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple containing batch of experiences and additional info
        """
        # Use all experiences if buffer size is less than batch_size
        if self.size < batch_size:
            indices = np.random.choice(self.size, batch_size, replace=True)
        else:
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities /= np.sum(probabilities)
            indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize weights
        
        # Increment beta gradually
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )
    
    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices of experiences to update
            errors: TD errors for each experience
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + self.epsilon


class TradingAgent:
    """Reinforcement learning agent for trading decisions."""
    
    def __init__(
        self,
        state_dim: int,
        action_space_type: str = 'discrete',
        action_size: int = 3,
        model_dir: str = 'saved_models',
        learning_rate: float = 0.0005,
        use_dueling: bool = True
    ):
        """
        Initialize the trading agent.
        
        Args:
            state_dim: Dimension of state space
            action_space_type: Type of action space ('discrete' or 'continuous')
            action_size: Size of action space (for discrete)
            model_dir: Directory to save models
            learning_rate: Learning rate for optimizer
            use_dueling: Whether to use dueling network architecture
        """
        self.state_dim = state_dim
        self.action_space_type = action_space_type
        self.action_size = action_size
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.use_dueling = use_dueling
        
        # Models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
    
    def _build_model(self) -> keras.Model:
        """
        Build the DQN model based on specified architecture.
        
        Returns:
            Keras model
        """
        input_layer = keras.layers.Input(shape=(self.state_dim,))
        norm = keras.layers.BatchNormalization()(input_layer)
        
        # Two branches for different feature extraction paths
        branch1 = keras.layers.Dense(256, activation='relu')(norm)
        branch1 = keras.layers.Dropout(0.3)(branch1)
        branch1 = keras.layers.Dense(128, activation='relu')(branch1)
        
        branch2 = keras.layers.Dense(64, activation='relu')(norm)
        
        merged = keras.layers.Concatenate()([branch1, branch2])
        dense = keras.layers.Dense(128, activation='relu')(merged)
        dense = keras.layers.Dropout(0.3)(dense)
        
        if self.use_dueling:
            # Dueling architecture (separate value and advantage streams)
            value_stream = keras.layers.Dense(64, activation='relu')(dense)
            value = keras.layers.Dense(1)(value_stream)
            
            if self.action_space_type == 'continuous':
                # For continuous action space
                advantage_stream = keras.layers.Dense(64, activation='relu')(dense)
                advantage = keras.layers.Dense(1)(advantage_stream)
                
                # Combine value and advantage
                output = keras.layers.Add()([value, advantage])
                # Use a custom activation layer instead of Lambda
                output = keras.layers.Activation('tanh')(output)
            else:
                # For discrete action space
                advantage_stream = keras.layers.Dense(64, activation='relu')(dense)
                advantage = keras.layers.Dense(self.action_size)(advantage_stream)
                
                # Instead of using Lambda, we'll explicitly implement the operation
                # Create a custom layer for this operation
                class AdvantageNormalization(keras.layers.Layer):
                    def __init__(self, **kwargs):
                        super(AdvantageNormalization, self).__init__(**kwargs)
                    
                    def call(self, inputs):
                        # Subtract mean from advantage
                        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
                        return inputs - mean
                    
                    def get_config(self):
                        config = super(AdvantageNormalization, self).get_config()
                        return config
                
                normalized_advantage = AdvantageNormalization()(advantage)
                output = keras.layers.Add()([value, normalized_advantage])
        else:
            # Standard architecture
            if self.action_space_type == 'continuous':
                output = keras.layers.Dense(1, activation='tanh')(dense)
            else:
                output = keras.layers.Dense(self.action_size, activation='linear')(dense)
        
        model = keras.Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=keras.losses.Huber()
        )
        
        return model
    
    def update_target_model(self) -> None:
        """Update target model weights from main model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> Union[int, float]:
        """
        Get action based on current state.
        
        Args:
            state: Current state
            epsilon: Exploration rate (0-1)
        
        Returns:
            Action to take
        """
        if np.random.rand() < epsilon:
            # Random action for exploration
            if self.action_space_type == 'continuous':
                return np.random.uniform(-1, 1)
            else:
                return np.random.choice(self.action_size)
        
        # Use model for exploitation
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        q_values = self.model(state_tensor)
        
        if self.action_space_type == 'continuous':
            return q_values[0, 0].numpy()
        else:
            return np.argmax(q_values[0])
    
    def train(
        self,
        replay_buffer: PrioritizedReplayBuffer,
        batch_size: int,
        gamma: float,
        use_double_dqn: bool = True
    ) -> float:
        """
        Train the agent using experiences from replay buffer.
        
        Args:
            replay_buffer: Prioritized replay buffer
            batch_size: Batch size for training
            gamma: Discount factor
            use_double_dqn: Whether to use Double DQN
        
        Returns:
            Loss value
        """
        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)
        
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            if self.action_space_type == 'continuous':
                # For continuous action space
                current_q = self.model(states_tensor)
                current_q_values = current_q[:, 0]
                
                if use_double_dqn:
                    next_q_main = self.model(next_states_tensor)
                    next_actions = tf.clip_by_value(next_q_main, -1, 1)
                    next_q_target = self.target_model(next_states_tensor)
                    next_q_values = next_q_target[:, 0]
                else:
                    next_q_values = self.target_model(next_states_tensor)[:, 0]
            else:
                # For discrete action space
                current_q = self.model(states_tensor)
                actions_tensor = tf.cast(actions, tf.int32)
                current_q_values = tf.reduce_sum(
                    current_q * tf.one_hot(actions_tensor, self.action_size),
                    axis=1
                )
                
                if use_double_dqn:
                    next_q_main = self.model(next_states_tensor)
                    next_actions = tf.argmax(next_q_main, axis=1)
                    next_q_target = self.target_model(next_states_tensor)
                    next_q_values = tf.reduce_sum(
                        next_q_target * tf.one_hot(next_actions, self.action_size),
                        axis=1
                    )
                else:
                    next_q_values = tf.reduce_max(self.target_model(next_states_tensor), axis=1)
            
            # Calculate target Q values
            target_q_values = rewards + gamma * next_q_values * (1 - dones)
            
            # TD errors for priority update
            td_errors = tf.abs(current_q_values - target_q_values).numpy()
            
            # Calculate loss with importance sampling weights
            loss = tf.reduce_mean(
                tf.keras.losses.huber(target_q_values, current_q_values) * weights
            )
        
        # Update replay buffer priorities
        replay_buffer.update_priorities(indices, td_errors)
        
        # Apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)  # Clip gradients
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss.numpy()
    
    def save_model(self, filename: str = 'trading_agent.keras') -> None:
        """
        Save the model to disk.
        
        Args:
            filename: Name of the file to save the model
        """
        path = self.model_dir / filename
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")
    
    def load_model(self, filename: str = 'trading_agent.keras') -> None:
        """
        Load the model from disk.
        
        Args:
            filename: Name of the file to load the model from
        """
        # Enable unsafe deserialization for Lambda layers
        try:
            # For TensorFlow 2.9+
            keras.config.enable_unsafe_deserialization()
            logger.info("Enabled unsafe deserialization for Lambda layers")
        except:
            # Fallback for older versions
            import tensorflow as tf
            try:
                tf.keras.utils.disable_interactive_logging()
            except:
                pass
        
        path = self.model_dir / filename
        if path.exists():
            try:
                self.model = keras.models.load_model(str(path))
                self.target_model.set_weights(self.model.get_weights())
                logger.info(f"Model loaded from {path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.info("Attempting alternative loading method...")
                try:
                    # Alternative loading method for older versions
                    self.model = keras.models.load_model(str(path), compile=False)
                    self.model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                        loss=keras.losses.Huber()
                    )
                    self.target_model.set_weights(self.model.get_weights())
                    logger.info(f"Model loaded with alternative method from {path}")
                except Exception as e2:
                    logger.error(f"Alternative loading also failed: {e2}")
        else:
            logger.warning(f"Model file not found at {path}")


#-------------------------------------------------------------------------
# Trading Environment Module
#-------------------------------------------------------------------------

@dataclass
class TradeInfo:
    """Information about a trade."""
    step: int
    type: str  # 'buy' or 'sell'
    amount: float
    price: float
    effective_price: float
    slippage: float
    fee: float
    original_action: Optional[Union[int, float]] = None
    modified_action: Optional[Union[int, float]] = None


class TradingEnvironment:
    """Advanced simulated trading environment for cryptocurrency trading."""
    
    def __init__(
        self,
        price_scaler: MinMaxScaler,
        data: np.ndarray,
        raw_data: pd.DataFrame,
        seq_length: int,
        initial_cash: float = 10000,
        max_position: float = 0.5,
        transaction_fee: float = 0.001,
        slippage_model: str = 'proportional',
        risk_management: bool = True,
        use_market_hours: bool = True,
        realistic_liquidity: bool = True,
        action_space_type: str = 'discrete',
        action_size: int = 3
    ):
        """
        Initialize the trading environment.
        
        Args:
            price_scaler: Scaler for price data
            data: Scaled feature data
            raw_data: Raw price data
            seq_length: Length of sequence for state representation
            initial_cash: Starting cash balance
            max_position: Maximum percentage of portfolio to hold in crypto
            transaction_fee: Transaction fee as a percentage
            slippage_model: Model for simulating slippage
            risk_management: Whether to apply risk management rules
            use_market_hours: Whether to consider market hours
            realistic_liquidity: Whether to consider realistic liquidity constraints
            action_space_type: Type of action space ('discrete' or 'continuous')
            action_size: Size of action space (for discrete)
        """
        self.price_scaler = price_scaler
        self.data = data
        self.raw_data = raw_data
        self.seq_length = seq_length
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.transaction_fee = transaction_fee
        self.slippage_model = slippage_model
        self.risk_management = risk_management
        self.use_market_hours = use_market_hours
        self.realistic_liquidity = realistic_liquidity
        self.action_space_type = action_space_type
        
        # Action space configuration
        if action_space_type == 'discrete':
            self.action_size = action_size
            self.discrete_actions = 11  # Enhanced action space: -100% to +100% in steps of 20%
        else:
            self.action_size = 1  # Continuous value between -1 and 1
        
        # Risk management parameters
        self.stop_loss_pct = 0.05  # 5% stop loss from entry price
        self.take_profit_pct = 0.10  # 10% take profit from entry price
        self.max_drawdown_pct = 0.20  # Maximum drawdown allowed before reducing position
        self.max_trade_size_pct = 0.10  # Maximum size of any single trade as % of portfolio
        
        # Trading frequency limits
        self.max_trades_per_day = 5
        self.trade_cooldown = 3  # Minimum steps between trades
        
        # Market parameters
        self.liquidity_factor = 0.5  # How much liquidity is available (0-1)
        
        # Initialize environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state
        """
        self.cash = self.initial_cash
        self.crypto_held = 0
        self.current_step = self.seq_length
        self.portfolio_value = self.cash
        self.max_portfolio_value = self.portfolio_value
        self.trade_history = []
        self.portfolio_values = [self.portfolio_value]
        self.trades_today = 0
        self.last_trade_day = 0
        self.last_trade_step = 0
        self.entry_prices = {}
        self.unrealized_pnl = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state representation.
        
        Returns:
            State vector
        """
        # Get price history
        price_history = self.data[self.current_step - self.seq_length:self.current_step]
        
        # Get current price
        current_price_scaled = self.data[self.current_step, 0]  # Scaled 'Close'
        current_price = self.price_scaler.inverse_transform([[current_price_scaled]])[0, 0]
        
        # Calculate position metrics
        portfolio_value = self.cash + self.crypto_held * current_price
        crypto_value = self.crypto_held * current_price
        crypto_position_pct = crypto_value / portfolio_value if portfolio_value > 0 else 0
        unrealized_pnl = portfolio_value - self.initial_cash
        pnl_pct = unrealized_pnl / self.initial_cash if self.initial_cash > 0 else 0
        
        # Technical indicators (from scaled data)
        price_change_pct = 0
        if self.current_step > 0:
            prev_price = self.price_scaler.inverse_transform([[self.data[self.current_step-1, 0]]])[0, 0]
            price_change_pct = (current_price / prev_price) - 1
        
        # Market state from additional features
        volatility = self.data[self.current_step, 8] if self.data.shape[1] > 8 else 0
        volume = self.data[self.current_step, 1] if self.data.shape[1] > 1 else 0
        
        # Trading restrictions
        days_since_last_trade = (self.current_step - self.last_trade_step) / 24  # Approx. for crypto
        trade_allowed = (self.trades_today < self.max_trades_per_day and 
                        (self.current_step - self.last_trade_step) >= self.trade_cooldown)
        
        # Concatenate all state information
        additional_features = np.array([
            current_price,
            self.cash,
            self.crypto_held,
            crypto_position_pct,
            pnl_pct,
            price_change_pct,
            volatility,
            volume,
            days_since_last_trade,
            float(trade_allowed)
        ])
        
        # Combine history with current state
        state = np.concatenate([
            price_history.flatten(),
            additional_features
        ])
        
        return state
    
    def calculate_slippage(self, action_size: float, current_price: float, volume: float) -> float:
        """
        Calculate slippage based on action size and market conditions.
        
        Args:
            action_size: Size of the action (as percentage)
            current_price: Current asset price
            volume: Trading volume
        
        Returns:
            Slippage as a percentage
        """
        if not self.realistic_liquidity:
            return 0
        
        # Base slippage on action size, volume, and volatility
        if self.slippage_model == 'proportional':
            # Proportional to order size and inversely proportional to volume
            return abs(action_size) * 0.001 * (1 / (volume + 1e-10))
        elif self.slippage_model == 'fixed':
            return 0.001  # 0.1% fixed slippage
        elif self.slippage_model == 'market_impact':
            # More sophisticated market impact model
            market_impact = abs(action_size) * 0.005 * (1 / (volume + 1e-10)) * current_price
            return market_impact / current_price  # Return as a percentage
        else:
            return 0
    
    def check_risk_management(
        self, 
        action: int, 
        crypto_value_pct: float, 
        current_price: float
    ) -> int:
        """
        Apply risk management rules to potentially modify action.
        
        Args:
            action: Original action
            crypto_value_pct: Percentage of portfolio in crypto
            current_price: Current asset price
        
        Returns:
            Potentially modified action
        """
        if not self.risk_management:
            return action
        
        # Check for stop loss / take profit for existing position
        if self.crypto_held > 0 and len(self.entry_prices) > 0:
            # Calculate average entry price
            total_value = sum(price * qty for price, qty in self.entry_prices.items())
            total_qty = sum(qty for _, qty in self.entry_prices.items())
            avg_entry = total_value / total_qty if total_qty > 0 else 0
            
            if avg_entry > 0:
                pnl_pct = (current_price / avg_entry) - 1
                
                # Stop loss hit
                if pnl_pct < -self.stop_loss_pct:
                    return 0  # Sell everything
                
                # Take profit hit
                if pnl_pct > self.take_profit_pct:
                    return 2  # Take partial profits
        
        # Check for maximum drawdown
        max_drawdown = 0
        if self.max_portfolio_value > 0:
            max_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            
        if max_drawdown > self.max_drawdown_pct:
            # Reduce position size when drawdown is too large
            if action > 5:  # Buy actions
                return 5  # Hold instead
            if self.crypto_held > 0:
                return 3  # Reduce position slightly
        
        # Check for maximum position size
        if crypto_value_pct > self.max_position and action > 5:
            return 5  # Hold instead of buying more
        
        # Limit trade size for any single trade
        if abs(action - 5) / 5 > self.max_trade_size_pct:
            # Scale back the action to match maximum trade size
            direction = 1 if action > 5 else -1
            return 5 + direction * int(5 * self.max_trade_size_pct)
        
        return action
    
    def step(self, action: Union[int, float]) -> Tuple[np.ndarray, float, bool]:
        """
        Execute a trading action and return the next state, reward, and done flag.
        
        Args:
            action: Action to take
        
        Returns:
            Tuple of (next_state, reward, done)
        """
        # Get current price and volume
        current_price_scaled = self.data[self.current_step, 0]  # Scaled 'Close'
        current_price = self.price_scaler.inverse_transform([[current_price_scaled]])[0, 0]
        
        volume_scaled = self.data[self.current_step, 1] if self.data.shape[1] > 1 else 0.5
        
        # Calculate current portfolio state
        crypto_value = self.crypto_held * current_price
        portfolio_value = self.cash + crypto_value
        crypto_value_pct = crypto_value / portfolio_value if portfolio_value > 0 else 0
        
        # Process action based on action space type
        if self.action_space_type == 'continuous':
            self._handle_continuous_action(action, current_price, volume_scaled)
        else:
            self._handle_discrete_action(action, current_price, volume_scaled, crypto_value_pct)
        
        # Move to next step
        self.current_step += 1
        
        # Reset daily trade counter if a new day begins
        current_day = self.current_step // 24  # Approx. daily periods for crypto
        if current_day > self.last_trade_day:
            self.trades_today = 0
            self.last_trade_day = current_day
        
        # Check if episode is done
        if self.current_step >= len(self.data) - 1:
            done = True
            reward = 0
            next_state = None
        else:
            # Get new price for next state
            next_price_scaled = self.data[self.current_step, 0]
            next_price = self.price_scaler.inverse_transform([[next_price_scaled]])[0, 0]
            
            # Calculate new portfolio value
            new_portfolio_value = self.cash + self.crypto_held * next_price
            self.portfolio_values.append(new_portfolio_value)
            
            # Update maximum portfolio value for drawdown calculation
            if new_portfolio_value > self.max_portfolio_value:
                self.max_portfolio_value = new_portfolio_value
            
            # Calculate reward
            reward = self._calculate_reward(new_portfolio_value)
            
            self.portfolio_value = new_portfolio_value
            done = False
            next_state = self._get_state()
        
        return next_state, reward, done
    
    def _handle_continuous_action(
        self, 
        action: float, 
        current_price: float, 
        volume_scaled: float
    ) -> None:
        """
        Handle continuous action space.
        
        Args:
            action: Action value between -1 and 1
            current_price: Current asset price
            volume_scaled: Scaled volume for slippage calculation
        """
        action_value = float(action)  # Should be between -1 and 1
        
        if action_value < -0.05:  # Sell
            sell_percentage = min(1.0, abs(action_value))
            sell_amount = self.crypto_held * sell_percentage
            
            # Calculate slippage for sell
            slippage_pct = self.calculate_slippage(sell_percentage, current_price, volume_scaled)
            effective_price = current_price * (1 - slippage_pct - self.transaction_fee)
            
            self.cash += sell_amount * effective_price
            self.crypto_held -= sell_amount
            
            # Record trade
            if sell_amount > 0:
                trade = TradeInfo(
                    step=self.current_step,
                    type='sell',
                    amount=sell_amount,
                    price=current_price,
                    effective_price=effective_price,
                    slippage=slippage_pct,
                    fee=self.transaction_fee * sell_amount * current_price
                )
                self.trade_history.append(trade)
                self.last_trade_step = self.current_step
                self.trades_today += 1
        
        elif action_value > 0.05:  # Buy
            buy_percentage = min(1.0, action_value)
            cash_to_spend = self.cash * buy_percentage
            
            # Calculate slippage for buy
            slippage_pct = self.calculate_slippage(buy_percentage, current_price, volume_scaled)
            effective_price = current_price * (1 + slippage_pct + self.transaction_fee)
            
            buy_amount = cash_to_spend / effective_price
            
            # Check position limits
            potential_crypto_value = (self.crypto_held + buy_amount) * current_price
            potential_position_pct = 0
            if self.cash - cash_to_spend + potential_crypto_value > 0:
                potential_position_pct = potential_crypto_value / (self.cash - cash_to_spend + potential_crypto_value)
            
            if potential_position_pct <= self.max_position:
                self.crypto_held += buy_amount
                self.cash -= cash_to_spend
                
                # Record entry price for this purchase
                self.entry_prices[effective_price] = buy_amount
                
                # Record trade
                trade = TradeInfo(
                    step=self.current_step,
                    type='buy',
                    amount=buy_amount,
                    price=current_price,
                    effective_price=effective_price,
                    slippage=slippage_pct,
                    fee=self.transaction_fee * cash_to_spend
                )
                self.trade_history.append(trade)
                self.last_trade_step = self.current_step
                self.trades_today += 1
    
    def _handle_discrete_action(
        self, 
        action: int, 
        current_price: float, 
        volume_scaled: float, 
        crypto_value_pct: float
    ) -> None:
        """
        Handle discrete action space.
        
        Args:
            action: Discrete action
            current_price: Current asset price
            volume_scaled: Scaled volume for slippage calculation
            crypto_value_pct: Percentage of portfolio in crypto
        """
        # Map the original discrete action space (0, 1, 2) to the enhanced one
        # Original: 0 = sell, 1 = hold, 2 = buy
        # Enhanced: 0-4 = sell (varying amounts), 5 = hold, 6-10 = buy (varying amounts)
        if isinstance(action, (int, np.integer)):
            if action == 0:  # Sell
                action = 0  # Sell 100%
            elif action == 1:  # Hold
                action = 5  # Hold
            elif action == 2:  # Buy
                action = 10  # Buy 100%
        
        # Apply risk management to potentially modify action
        original_action = action
        action = self.check_risk_management(action, crypto_value_pct, current_price)
        
        # Map discrete action (0-10) to percentage (-100% to +100%)
        action_percentage = (action - 5) / 5  # Converts 0->-100%, 5->0%, 10->100%
        
        if action_percentage < -0.05:  # Sell
            sell_percentage = min(1.0, abs(action_percentage))
            sell_amount = self.crypto_held * sell_percentage
            
            # Calculate slippage for sell
            slippage_pct = self.calculate_slippage(sell_percentage, current_price, volume_scaled)
            effective_price = current_price * (1 - slippage_pct - self.transaction_fee)
            
            self.cash += sell_amount * effective_price
            self.crypto_held -= sell_amount
            
            # Record trade
            if sell_amount > 0:
                trade = TradeInfo(
                    step=self.current_step,
                    type='sell',
                    amount=sell_amount,
                    price=current_price,
                    effective_price=effective_price,
                    slippage=slippage_pct,
                    fee=self.transaction_fee * sell_amount * current_price,
                    original_action=original_action,
                    modified_action=action
                )
                self.trade_history.append(trade)
                self.last_trade_step = self.current_step
                self.trades_today += 1
        
        elif action_percentage > 0.05:  # Buy
            buy_percentage = min(1.0, action_percentage)
            cash_to_spend = self.cash * buy_percentage
            
            # Calculate slippage for buy
            slippage_pct = self.calculate_slippage(buy_percentage, current_price, volume_scaled)
            effective_price = current_price * (1 + slippage_pct + self.transaction_fee)
            
            buy_amount = cash_to_spend / effective_price
            
            # Check position limits
            potential_crypto_value = (self.crypto_held + buy_amount) * current_price
            portfolio_after_buy = self.cash - cash_to_spend + potential_crypto_value
            potential_position_pct = potential_crypto_value / portfolio_after_buy if portfolio_after_buy > 0 else 0
            
            if potential_position_pct <= self.max_position:
                self.crypto_held += buy_amount
                self.cash -= cash_to_spend
                
                # Record entry price for this purchase
                self.entry_prices[effective_price] = buy_amount
                
                # Record trade
                trade = TradeInfo(
                    step=self.current_step,
                    type='buy',
                    amount=buy_amount,
                    price=current_price,
                    effective_price=effective_price,
                    slippage=slippage_pct,
                    fee=self.transaction_fee * cash_to_spend,
                    original_action=original_action,
                    modified_action=action
                )
                self.trade_history.append(trade)
                self.last_trade_step = self.current_step
                self.trades_today += 1
    
    def _calculate_reward(self, new_portfolio_value: float) -> float:
        """
        Calculate reward based on portfolio performance and risk metrics.
        
        Args:
            new_portfolio_value: Updated portfolio value
        
        Returns:
            Calculated reward
        """
        # Simple return component
        simple_return = new_portfolio_value - self.portfolio_value
        
        # Sharpe-like ratio component (reward / risk)
        returns_history = np.diff(self.portfolio_values[-20:]) if len(self.portfolio_values) > 20 else np.diff(self.portfolio_values)
        stddev = np.std(returns_history) if len(returns_history) > 0 else 1
        sharpe_component = simple_return / (stddev + 1e-9)  # Avoid division by zero
        
        # Drawdown penalty
        drawdown = 0
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - new_portfolio_value) / self.max_portfolio_value
        
        drawdown_penalty = -100 * drawdown if drawdown > self.max_drawdown_pct else 0
        
        # Inactivity penalty - small penalty for not trading for long periods
        inactivity_factor = max(0, (self.current_step - self.last_trade_step)) / 100
        inactivity_penalty = -0.1 * inactivity_factor if inactivity_factor > 1 else 0
        
        # Combined reward
        reward = simple_return + 0.1 * sharpe_component + drawdown_penalty + inactivity_penalty
        
        return reward
    
    def render(self, mode: str = 'human') -> None:
        """
        Visualize the current state of the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            # Current portfolio state
            current_price_scaled = self.data[self.current_step - 1, 0]
            current_price = self.price_scaler.inverse_transform([[current_price_scaled]])[0, 0]
            crypto_value = self.crypto_held * current_price
            portfolio_value = self.cash + crypto_value
            
            print(f"\n=== Step {self.current_step} ===")
            print(f"Cash: ${self.cash:.2f}")
            print(f"Crypto: {self.crypto_held:.8f} (${crypto_value:.2f})")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(f"Initial Cash: ${self.initial_cash:.2f}")
            print(f"Return: {((portfolio_value / self.initial_cash) - 1) * 100:.2f}%")
            
            # Calculate drawdown
            max_drawdown = 0
            if self.max_portfolio_value > 0:
                max_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
            
            print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
            
            # Recent trades
            if len(self.trade_history) > 0:
                recent_trades = self.trade_history[-3:]
                print("\nRecent Trades:")
                for trade in recent_trades:
                    trade_type = trade.type.upper()
                    print(f"{trade_type} {trade.amount:.8f} at ${trade.price:.2f} " +
                        f"(effective: ${trade.effective_price:.2f})")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate and return key performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if len(self.portfolio_values) < 2:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_profit_per_trade': 0
            }
        
        # Calculate returns
        total_return = (self.portfolio_values[-1] / self.portfolio_values[0]) - 1
        daily_returns = []
        
        # Approximate daily returns (assuming steps are in sequence)
        for i in range(1, len(self.portfolio_values)):
            daily_returns.append(self.portfolio_values[i] / self.portfolio_values[i-1] - 1)
        
        # Calculate Sharpe ratio (annualized)
        daily_returns_array = np.array(daily_returns)
        avg_return = np.mean(daily_returns_array)
        std_return = np.std(daily_returns_array) + 1e-9  # Avoid division by zero
        sharpe_ratio = avg_return / std_return * np.sqrt(365)
        
        # Calculate maximum drawdown
        max_drawdown = 0
        peak = self.portfolio_values[0]
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate trade statistics
        num_trades = len(self.trade_history)
        profitable_trades = 0
        total_profit = 0
        
        # Match buys and sells to calculate profit/loss
        positions = {}
        
        for trade in self.trade_history:
            if trade.type == 'buy':
                positions[trade.step] = {
                    'price': trade.effective_price,
                    'amount': trade.amount
                }
            elif trade.type == 'sell':
                # Find oldest open position
                sell_amount_remaining = trade.amount
                sell_price = trade.effective_price
                sell_value = 0
                buy_value = 0
                
                for step, position in sorted(positions.items()):
                    if position['amount'] > 0:
                        amount_from_position = min(position['amount'], sell_amount_remaining)
                        position['amount'] -= amount_from_position
                        sell_amount_remaining -= amount_from_position
                        
                        # Calculate P&L for this portion
                        buy_value += amount_from_position * position['price']
                        sell_value += amount_from_position * sell_price
                        
                        if sell_amount_remaining <= 0:
                            break
                
                if buy_value > 0:
                    trade_profit = sell_value - buy_value
                    total_profit += trade_profit
                    if trade_profit > 0:
                        profitable_trades += 1
        
        win_rate = profitable_trades / num_trades if num_trades > 0 else 0
        avg_profit_per_trade = total_profit / num_trades if num_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit_per_trade
        }


#-------------------------------------------------------------------------
# Trading System Module
#-------------------------------------------------------------------------

class CryptoTradingSystem:
    """Main trading system that integrates all components."""
    
    def __init__(
        self,
        ticker: str = 'TSLA',
        start_date: str = '2020-01-01',
        end_date: str = '2024-03-15',
        seq_length: int = 60,
        initial_cash: float = 10000,
        model_dir: str = 'saved_models',
        action_space_type: str = 'discrete',
        use_gpu: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the trading system.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date for data
            end_date: End date for data
            seq_length: Sequence length for time series
            initial_cash: Initial cash for trading
            model_dir: Directory to save models
            action_space_type: Type of action space ('discrete' or 'continuous')
            use_gpu: Whether to use GPU for training
            verbose: Whether to print verbose output
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.seq_length = seq_length
        self.initial_cash = initial_cash
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.action_space_type = action_space_type
        self.verbose = verbose
        
        # Configure GPU
        if use_gpu:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                logger.info("Using GPU for training")
            else:
                logger.info("No GPU found, using CPU instead")
        
        # Initialize components
        self.data_processor = DataProcessor(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            seq_length=seq_length
        )
        
        # Download and prepare data
        self.X, self.y = self.data_processor.prepare_data()
        
        # Initialize price predictor
        self.price_predictor = PricePredictor(
            seq_length=seq_length,
            num_features=self.X.shape[2],
            model_dir=model_dir
        )
        
        # Initialize trading environment
        self.env = TradingEnvironment(
            price_scaler=self.data_processor.price_scaler,
            data=self.data_processor.scaled_features,
            raw_data=self.data_processor.raw_data,
            seq_length=seq_length,
            initial_cash=initial_cash,
            action_space_type=action_space_type
        )
        
        # Calculate state dimension for the agent
        state = self.env.reset()
        state_dim = state.shape[0]
        
        # Initialize trading agent
        self.agent = TradingAgent(
            state_dim=state_dim,
            action_space_type=action_space_type,
            action_size=3 if action_space_type == 'discrete' else 1,
            model_dir=model_dir,
            use_dueling=True
        )
    
    def train_predictive_models(self, epochs: int = 10) -> None:
        """
        Train price prediction models.
        
        Args:
            epochs: Number of training epochs
        """
        logger.info("Training price prediction models...")
        self.price_predictor.train_models(self.X, self.y, epochs=epochs)
        logger.info("Price prediction models trained successfully")
    
    def train_agent(
        self,
        num_episodes: int = 1000,
        batch_size: int = 64,
        gamma: float = 0.99,
        target_update_freq: int = 10,
        save_freq: int = 50,
        eval_freq: int = 20,
        learning_starts: int = 1000,
        double_dqn: bool = True
    ) -> float:
        """
        Train the trading agent.
        
        Args:
            num_episodes: Number of training episodes
            batch_size: Batch size for training
            gamma: Discount factor
            target_update_freq: Frequency of target network updates
            save_freq: Frequency of model saving
            eval_freq: Frequency of evaluation
            learning_starts: Number of steps before starting training
            double_dqn: Whether to use Double DQN
            
        Returns:
            Best evaluation reward
        """
        logger.info("Training trading agent...")
        
        # Initialize replay buffer
        buffer_size = 100000
        replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        
        # Epsilon-greedy exploration parameters
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        epsilon = epsilon_start
        
        total_steps = 0
        best_reward = float('-inf')
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Choose action
                if np.random.rand() < epsilon or total_steps < learning_starts:
                    if self.action_space_type == 'continuous':
                        action = np.random.uniform(-1, 1)
                    else:
                        action = np.random.choice(self.agent.action_size)
                else:
                    action = self.agent.get_action(state)
                
                # Take action in environment
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                
                # Store experience in replay buffer
                if next_state is not None:
                    replay_buffer.add(state, action, reward, next_state, done)
                    
                    # Train agent
                    if total_steps >= learning_starts and total_steps % 4 == 0:
                        loss = self.agent.train(replay_buffer, batch_size, gamma, double_dqn)
                    
                    # Update target network
                    if total_steps % (target_update_freq * batch_size) == 0:
                        self.agent.update_target_model()
                    
                    state = next_state
                    total_steps += 1
            
            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # Track rewards
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])
            
            if self.verbose:
                logger.info(f"Episode {episode}/{num_episodes} | Reward: {episode_reward:.2f} | " +
                          f"Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.4f}")
            
            # Evaluate agent periodically
            if episode % eval_freq == 0:
                eval_reward = self.evaluate_agent(num_episodes=5)
                logger.info(f"Evaluation at episode {episode}: Average Reward = {eval_reward:.2f}")
                
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self.agent.save_model('best_trading_agent.keras')
                    logger.info(f"Saved new best model with reward {best_reward:.2f}")
            
            # Save model periodically
            if episode % save_freq == 0:
                self.agent.save_model(f'trading_agent_episode_{episode}.keras')
        
        # Save final model
        self.agent.save_model('final_trading_agent.keras')
        
        # Save scalers for later use
        self.data_processor.save_scalers(self.model_dir)
        
        logger.info(f"Trading agent training completed. Best evaluation reward: {best_reward:.2f}")
        
        # Load the best model for future use
        self.agent.load_model('best_trading_agent.keras')
        
        return best_reward
    
    def evaluate_agent(self, num_episodes: int = 10) -> float:
        """
        Evaluate the agent's performance.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Average reward across episodes
        """
        total_rewards = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.agent.get_action(state, epsilon=0.0)  # No exploration
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                
                if next_state is not None:
                    state = next_state
                else:
                    break
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def load_models(self) -> None:
        """Load all trained models and components."""
        # Load price predictor models
        self.price_predictor.load_models()
        
        # Load trading agent
        self.agent.load_model('best_trading_agent.keras')
        
        # Load scalers
        self.data_processor.load_scalers(self.model_dir)
        
        logger.info("All models loaded successfully")
    
    def predict_price(self, input_sequence: np.ndarray) -> float:
        """
        Predict future price using the trained models.
        
        Args:
            input_sequence: Input sequence
            
        Returns:
            Predicted price
        """
        scaled_prediction = self.price_predictor.predict_price(input_sequence)
        
        if scaled_prediction is not None:
            # Convert from scaled prediction to actual price
            actual_price = self.data_processor.price_scaler.inverse_transform([[scaled_prediction]])[0, 0]
            return actual_price
        
        return None
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run a backtest of the trading strategy.
        
        Returns:
            Dictionary of backtest results
        """
        logger.info("Running backtest...")
        
        state = self.env.reset()
        done = False
        
        buy_count = 0
        sell_count = 0
        hold_count = 0
        
        initial_portfolio = self.env.portfolio_value
        
        while not done:
            # Get action from agent
            action = self.agent.get_action(state, epsilon=0.0)  # No exploration during testing
            
            # Count action types
            if self.action_space_type == 'continuous':
                if action < -0.05:
                    sell_count += 1
                elif action > 0.05:
                    buy_count += 1
                else:
                    hold_count += 1
            else:
                if action == 0:
                    sell_count += 1
                elif action == 1:
                    hold_count += 1
                else:
                    buy_count += 1
            
            # Take action in environment
            next_state, reward, done = self.env.step(action)
            
            if not done:
                state = next_state
            else:
                break
        
        # Get final results
        final_portfolio = self.env.portfolio_value
        performance_metrics = self.env.get_performance_metrics()
        
        # Calculate buy-and-hold performance for comparison
        first_price = self.data_processor.price_scaler.inverse_transform(
            [[self.data_processor.scaled_features[self.seq_length, 0]]]
        )[0, 0]
        
        last_price = self.data_processor.price_scaler.inverse_transform(
            [[self.data_processor.scaled_features[-1, 0]]]
        )[0, 0]
        
        initial_crypto = self.initial_cash / first_price
        buyhold_value = initial_crypto * last_price
        
        results = {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
            "initial_value": initial_portfolio,
            "final_value": final_portfolio,
            "gain_pct": ((final_portfolio / initial_portfolio) - 1) * 100,
            "sharpe_ratio": performance_metrics['sharpe_ratio'],
            "max_drawdown": performance_metrics['max_drawdown'],
            "win_rate": performance_metrics['win_rate'],
            "total_trades": performance_metrics['num_trades'],
            "buyhold_value": buyhold_value,
            "vs_buyhold_pct": ((final_portfolio / buyhold_value) - 1) * 100,
            "trade_history": self.env.trade_history
        }
        
        return results
    
    def save_backtest_results(self, results: Dict[str, Any], filename: str = 'backtest_results.json') -> None:
        """
        Save backtest results to file.
        
        Args:
            results: Dictionary of backtest results
            filename: Filename to save results
        """
        path = self.model_dir / filename
        
        # Convert trade history to serializable format
        serializable_results = results.copy()
        
        if 'trade_history' in serializable_results:
            serializable_results['trade_history'] = [
                {
                    'step': trade.step,
                    'type': trade.type,
                    'amount': float(trade.amount),
                    'price': float(trade.price),
                    'effective_price': float(trade.effective_price),
                    'slippage': float(trade.slippage),
                    'fee': float(trade.fee)
                }
                for trade in serializable_results['trade_history']
            ]
        
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Backtest results saved to {path}")


#-------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------

def main():
    """Main function to run the trading system."""
    try:
        logger.info("Initializing Crypto Trading System...")
        
        # Create trading system (can switch between discrete and continuous action spaces)
        trading_system = CryptoTradingSystem(
            ticker='TSLA',
            start_date='2024-01-01',
            end_date='2025-03-11',
            seq_length=60,
            initial_cash=10000,
            action_space_type='discrete',  # 'discrete' or 'continuous'
            verbose=True
        )
        
        # Train price prediction models
        trading_system.train_predictive_models(epochs=50)
        
        # Train trading agent
        trading_system.train_agent(
            num_episodes=200,  # Increase for better results (e.g., 100, 500, 1000)
            double_dqn=True,
            batch_size=128,
            gamma=0.99,
            target_update_freq=10,
            save_freq=20
        )
        
        # Run backtest
        results = trading_system.run_backtest()
        
        # Save backtest results
        trading_system.save_backtest_results(results)
        
        # Print backtest results
        logger.info("\nBacktest Results:")
        logger.info(f"  Buy actions: {results['buy_count']}")
        logger.info(f"  Hold actions: {results['hold_count']}")
        logger.info(f"  Sell actions: {results['sell_count']}")
        logger.info(f"  Total trades: {results['total_trades']}")
        logger.info(f"  Win rate: {results['win_rate']:.2f}")
        logger.info(f"  Initial portfolio: ${results['initial_value']:.2f}")
        logger.info(f"  Final portfolio: ${results['final_value']:.2f}")
        logger.info(f"  Performance: {results['gain_pct']:.2f}%")
        logger.info(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"  Maximum drawdown: {results['max_drawdown']:.2f}")
        logger.info(f"  Buy and hold value: ${results['buyhold_value']:.2f}")
        logger.info(f"  vs Buy-hold: {results['vs_buyhold_pct']:.2f}%")
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)


if __name__ == "__main__":
    main()