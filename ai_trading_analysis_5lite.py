import os
import json
import logging
import random
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crypto_trading_lite')

# Constants
TRANSACTION_FEE = 0.001  # 0.1% transaction fee

#-------------------------------------------------------------------------
# Data Module - Simplified
#-------------------------------------------------------------------------

class DataProcessor:
    """Handles data downloading, preprocessing, and feature engineering."""
    
    def __init__(
        self, 
        ticker: str = 'TSLA', 
        start_date: str = '2020-01-01', 
        end_date: str = '2024-03-15',
        seq_length: int = 30,  # Reduced from 60
        random_state: int = 42
    ):
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
        self.X = None
        self.y = None
    
    def download_data(self) -> pd.DataFrame:
        """Download historical data for the specified ticker."""
        logger.info(f"Downloading data for {self.ticker} from {self.start_date} to {self.end_date}")
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        if len(data) < self.seq_length + 1:
            raise ValueError(f"Insufficient data. Need at least {self.seq_length + 1} data points.")
        
        self.raw_data = data
        return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add simplified technical indicators to the dataset."""
        df = data.copy()
        
        # Simple moving averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        
        # RSI (simplified)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean().replace(0, 0.001)
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        
        # Fill NaN values
        df = df.ffill().fillna(0)
        return df
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and preprocess financial data with simplified features."""
        if self.raw_data is None:
            self.download_data()
        
        # Add technical indicators
        self.processed_data = self.add_technical_indicators(self.raw_data)
        
        # Select fewer features for the model
        feature_columns = [
            'Close', 'Volume', 'MA_7', 'MA_21', 'RSI', 'Price_Change'
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
        """Create sequences for time series prediction."""
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
# Model Module - Simplified
#-------------------------------------------------------------------------

class PricePredictor:
    """Handles simplified price prediction models."""
    
    def __init__(
        self, 
        seq_length: int, 
        num_features: int,
        model_dir: str = 'saved_models'
    ):
        self.seq_length = seq_length
        self.num_features = num_features
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Just one model for simplicity
        self.model = None
    
    def build_model(self) -> keras.Model:
        """Build a simplified model for price prediction."""
        input_layer = keras.layers.Input(shape=(self.seq_length, self.num_features))
        
        # Simple convolutional model
        conv1 = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(input_layer)
        pool1 = keras.layers.GlobalAveragePooling1D()(conv1)
        
        # Output layers
        dense1 = keras.layers.Dense(64, activation='relu')(pool1)
        dropout1 = keras.layers.Dropout(0.2)(dense1)
        output = keras.layers.Dense(1, activation='linear')(dropout1)
        
        # Create and compile model
        model = keras.Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Higher learning rate
            loss=keras.losses.MeanSquaredError()  # MSE for simplicity
        )
        
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 20) -> None:
        """Train price prediction model with fewer epochs."""
        # Build model
        self.model = self.build_model()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        logger.info("Training price prediction model...")
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(self.model_dir / 'best_price_model.keras'),
            save_best_only=True,
            monitor='val_loss'
        )
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        # Load best model
        self.model = keras.models.load_model(str(self.model_dir / 'best_price_model.keras'))
        logger.info("Price prediction model trained successfully")
    
    def load_model(self) -> None:
        """Load pre-trained model from disk."""
        model_path = self.model_dir / 'best_price_model.keras'
        
        if model_path.exists():
            try:
                self.model = keras.models.load_model(str(model_path))
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        else:
            logger.warning(f"Model not found at {model_path}")
    
    def predict_price(self, input_sequence: np.ndarray) -> float:
        """Predict future price using the model."""
        if input_sequence.ndim == 2:
            input_sequence = input_sequence[np.newaxis, :, :]
        
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train_model() or load_model() first.")
        
        try:
            prediction = self.model.predict(input_sequence, verbose=0)[0, 0]
            return prediction
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return None


#-------------------------------------------------------------------------
# Simple Replay Buffer
#-------------------------------------------------------------------------

class SimpleReplayBuffer:
    """Simple experience replay buffer for more efficient learning."""
    
    def __init__(self, capacity: int):
        """Initialize the simple replay buffer."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(
        self, 
        state: np.ndarray,
        action: Union[int, float],
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a new experience to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences randomly."""
        # Adjust batch size if buffer is not filled enough
        actual_batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, actual_batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )


#-------------------------------------------------------------------------
# Trading Agent - Simplified
#-------------------------------------------------------------------------

class TradingAgent:
    """Simplified reinforcement learning agent for trading decisions."""
    
    def __init__(
        self,
        state_dim: int,
        action_space_type: str = 'discrete',
        action_size: int = 3,
        model_dir: str = 'saved_models',
        learning_rate: float = 0.001  # Higher learning rate
    ):
        self.state_dim = state_dim
        self.action_space_type = action_space_type
        self.action_size = action_size
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        
        # Models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
    
    def _build_model(self) -> keras.Model:
        """Build a simplified DQN model for faster learning."""
        input_layer = keras.layers.Input(shape=(self.state_dim,))
        
        # Simple architecture with fewer layers
        dense1 = keras.layers.Dense(64, activation='relu')(input_layer)
        dense2 = keras.layers.Dense(32, activation='relu')(dense1)
        
        if self.action_space_type == 'continuous':
            output = keras.layers.Dense(1, activation='tanh')(dense2)
        else:
            output = keras.layers.Dense(self.action_size, activation='linear')(dense2)
        
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
        """Get action based on current state."""
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
        replay_buffer: SimpleReplayBuffer,
        batch_size: int,
        gamma: float
    ) -> float:
        """Train the agent using experiences from replay buffer."""
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            if self.action_space_type == 'continuous':
                # For continuous action space
                current_q = self.model(states_tensor)
                current_q_values = current_q[:, 0]
                
                next_q_values = self.target_model(next_states_tensor)[:, 0]
            else:
                # For discrete action space
                current_q = self.model(states_tensor)
                actions_tensor = tf.cast(actions, tf.int32)
                current_q_values = tf.reduce_sum(
                    current_q * tf.one_hot(actions_tensor, self.action_size),
                    axis=1
                )
                
                next_q_values = tf.reduce_max(self.target_model(next_states_tensor), axis=1)
            
            # Calculate target Q values
            target_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)
            
            # Calculate loss (simplified)
            loss = tf.keras.losses.huber(target_q_values, current_q_values)
            loss = tf.reduce_mean(loss)
        
        # Apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss.numpy()
    
    def save_model(self, filename: str = 'trading_agent.keras') -> None:
        """Save the model to disk."""
        path = self.model_dir / filename
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")
    
    def load_model(self, filename: str = 'trading_agent.keras') -> None:
        """Load the model from disk."""
        path = self.model_dir / filename
        if path.exists():
            try:
                self.model = keras.models.load_model(str(path))
                self.target_model.set_weights(self.model.get_weights())
                logger.info(f"Model loaded from {path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        else:
            logger.warning(f"Model file not found at {path}")


#-------------------------------------------------------------------------
# Trade Info
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


#-------------------------------------------------------------------------
# Trading Environment - Simplified
#-------------------------------------------------------------------------

class TradingEnvironment:
    """Simplified trading environment for faster learning."""
    
    def __init__(
        self,
        price_scaler: MinMaxScaler,
        data: np.ndarray,
        raw_data: pd.DataFrame,
        seq_length: int,
        initial_cash: float = 10000,
        max_position: float = 0.5,
        transaction_fee: float = 0.001,
        action_space_type: str = 'discrete',
        action_size: int = 3
    ):
        self.price_scaler = price_scaler
        self.data = data
        self.raw_data = raw_data
        self.seq_length = seq_length
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.transaction_fee = transaction_fee
        self.action_space_type = action_space_type
        self.action_size = action_size
        
        # Initialize environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.cash = self.initial_cash
        self.crypto_held = 0
        self.current_step = self.seq_length
        self.portfolio_value = self.cash
        self.max_portfolio_value = self.portfolio_value
        self.trade_history = []
        self.portfolio_values = [self.portfolio_value]
        self.last_trade_step = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get the current state representation."""
        # Get price history
        price_history = self.data[self.current_step - self.seq_length:self.current_step]
        
        # Get current price
        current_price_scaled = self.data[self.current_step, 0]  # Scaled 'Close'
        current_price = self.price_scaler.inverse_transform([[current_price_scaled]])[0, 0]
        
        # Calculate position metrics
        portfolio_value = self.cash + self.crypto_held * current_price
        crypto_value = self.crypto_held * current_price
        crypto_position_pct = crypto_value / portfolio_value if portfolio_value > 0 else 0
        
        # Simplified additional features
        additional_features = np.array([
            current_price,
            self.cash,
            self.crypto_held,
            crypto_position_pct,
            float(self.current_step - self.last_trade_step)  # Steps since last trade
        ])
        
        # Combine history with current state
        state = np.concatenate([
            price_history.flatten(),
            additional_features
        ])
        
        return state
    
    def _calculate_reward(self, new_portfolio_value: float) -> float:
        """
        Calculate a simplified reward based on portfolio performance.
        
        This reward function focuses on percentage returns rather than absolute values,
        which provides clearer learning signals for the agent.
        """
        # Calculate percentage return (rather than absolute)
        pct_return = (new_portfolio_value / self.portfolio_value) - 1.0
        
        # Scale the percentage return to make it a stronger signal
        scaled_return = pct_return * 100
        
        # Small bonus for successful trades (to encourage exploration)
        trade_bonus = 0
        if len(self.trade_history) > 0 and self.trade_history[-1].step == self.current_step:
            # Add a small bonus just for making a trade to encourage exploration
            trade_bonus = 0.1
        
        # Simplified reward
        reward = scaled_return + trade_bonus
        
        return reward
    
    def step(self, action: Union[int, float]) -> Tuple[np.ndarray, float, bool]:
        """Execute a trading action and return the next state, reward, and done flag."""
        # Get current price
        current_price_scaled = self.data[self.current_step, 0]
        current_price = self.price_scaler.inverse_transform([[current_price_scaled]])[0, 0]
        
        # Calculate current portfolio state
        crypto_value = self.crypto_held * current_price
        portfolio_value = self.cash + crypto_value
        
        # Process action based on action space type
        if self.action_space_type == 'continuous':
            self._handle_continuous_action(action, current_price)
        else:
            self._handle_discrete_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
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
    
    def _handle_continuous_action(self, action: float, current_price: float) -> None:
        """Handle continuous action space."""
        action_value = float(action)  # Should be between -1 and 1
        
        if action_value < -0.05:  # Sell
            sell_percentage = min(1.0, abs(action_value))
            sell_amount = self.crypto_held * sell_percentage
            
            effective_price = current_price * (1 - self.transaction_fee)
            
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
                    slippage=0,
                    fee=self.transaction_fee * sell_amount * current_price
                )
                self.trade_history.append(trade)
                self.last_trade_step = self.current_step
        
        elif action_value > 0.05:  # Buy
            buy_percentage = min(1.0, action_value)
            cash_to_spend = self.cash * buy_percentage
            
            effective_price = current_price * (1 + self.transaction_fee)
            
            buy_amount = cash_to_spend / effective_price
            
            # Simple position limit check
            potential_crypto_value = (self.crypto_held + buy_amount) * current_price
            potential_position_pct = 0
            if self.cash - cash_to_spend + potential_crypto_value > 0:
                potential_position_pct = potential_crypto_value / (self.cash - cash_to_spend + potential_crypto_value)
            
            if potential_position_pct <= self.max_position:
                self.crypto_held += buy_amount
                self.cash -= cash_to_spend
                
                # Record trade
                trade = TradeInfo(
                    step=self.current_step,
                    type='buy',
                    amount=buy_amount,
                    price=current_price,
                    effective_price=effective_price,
                    slippage=0,
                    fee=self.transaction_fee * cash_to_spend
                )
                self.trade_history.append(trade)
                self.last_trade_step = self.current_step
    
    def _handle_discrete_action(self, action: int, current_price: float) -> None:
        """Handle discrete action space."""
        if action == 0:  # Sell all
            sell_amount = self.crypto_held
            
            if sell_amount > 0:
                effective_price = current_price * (1 - self.transaction_fee)
                
                self.cash += sell_amount * effective_price
                self.crypto_held = 0
                
                # Record trade
                trade = TradeInfo(
                    step=self.current_step,
                    type='sell',
                    amount=sell_amount,
                    price=current_price,
                    effective_price=effective_price,
                    slippage=0,
                    fee=self.transaction_fee * sell_amount * current_price
                )
                self.trade_history.append(trade)
                self.last_trade_step = self.current_step
        
        elif action == 2:  # Buy with 50% of available cash
            buy_percentage = 0.5
            cash_to_spend = self.cash * buy_percentage
            
            if cash_to_spend > 0:
                effective_price = current_price * (1 + self.transaction_fee)
                
                buy_amount = cash_to_spend / effective_price
                
                # Simple position limit check
                potential_crypto_value = (self.crypto_held + buy_amount) * current_price
                potential_position_pct = 0
                if self.cash - cash_to_spend + potential_crypto_value > 0:
                    potential_position_pct = potential_crypto_value / (self.cash - cash_to_spend + potential_crypto_value)
                
                if potential_position_pct <= self.max_position:
                    self.crypto_held += buy_amount
                    self.cash -= cash_to_spend
                    
                    # Record trade
                    trade = TradeInfo(
                        step=self.current_step,
                        type='buy',
                        amount=buy_amount,
                        price=current_price,
                        effective_price=effective_price,
                        slippage=0,
                        fee=self.transaction_fee * cash_to_spend
                    )
                    self.trade_history.append(trade)
                    self.last_trade_step = self.current_step
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate and return key performance metrics."""
        if len(self.portfolio_values) < 2:
            return {
                'total_return': 0,
                'max_drawdown': 0,
                'num_trades': 0,
            }
        
        # Calculate returns
        total_return = (self.portfolio_values[-1] / self.portfolio_values[0]) - 1
        
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
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
        }


#-------------------------------------------------------------------------
# Trading System Module - Simplified
#-------------------------------------------------------------------------

class CryptoTradingSystemLite:
    """Simplified trading system for faster learning."""
    
    def __init__(
        self,
        ticker: str = 'TSLA',
        start_date: str = '2020-01-01',
        end_date: str = '2024-03-15',
        seq_length: int = 30,  # Reduced from 60
        initial_cash: float = 10000,
        model_dir: str = 'saved_models',
        action_space_type: str = 'discrete',
        verbose: bool = True
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.seq_length = seq_length
        self.initial_cash = initial_cash
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.action_space_type = action_space_type
        self.verbose = verbose
        
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
            action_space_type=action_space_type,
            action_size=3 if action_space_type == 'discrete' else 1
        )
        
        # Calculate state dimension for the agent
        state = self.env.reset()
        state_dim = state.shape[0]
        
        # Initialize trading agent
        self.agent = TradingAgent(
            state_dim=state_dim,
            action_space_type=action_space_type,
            action_size=3 if action_space_type == 'discrete' else 1,
            model_dir=model_dir
        )
    
    def train_predictive_model(self, epochs: int = 20) -> None:
        """Train price prediction model with fewer epochs."""
        logger.info("Training price prediction model...")
        self.price_predictor.train_model(self.X, self.y, epochs=epochs)
        logger.info("Price prediction model trained successfully")
    
    def train_agent(
        self,
        num_episodes: int = 200,  # Reduced number of episodes
        batch_size: int = 32,     # Smaller batch size for faster updates
        gamma: float = 0.95,      # Slightly reduced discount factor for more immediate rewards
        target_update_freq: int = 5,  # More frequent target updates
        save_freq: int = 25,      # Save less frequently
        eval_freq: int = 10,      # Evaluate more frequently
        learning_starts: int = 100  # Start learning earlier
    ) -> float:
        """Train the trading agent with simplified parameters for faster learning."""
        logger.info("Training trading agent...")
        
        # Initialize replay buffer (simple version)
        buffer_size = 10000  # Smaller buffer
        replay_buffer = SimpleReplayBuffer(capacity=buffer_size)
        
        # Epsilon-greedy exploration parameters
        epsilon_start = 1.0
        epsilon_end = 0.1  # Higher end value for more exploration
        epsilon_decay = 0.98  # Faster decay
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
                    
                    # Train agent (more frequent updates)
                    if total_steps >= learning_starts and total_steps % 2 == 0:
                        loss = self.agent.train(replay_buffer, batch_size, gamma)
                    
                    # Update target network more frequently
                    if total_steps % (target_update_freq * batch_size) == 0:
                        self.agent.update_target_model()
                    
                    state = next_state
                    total_steps += 1
            
            # Decay epsilon more aggressively at the beginning
            if episode < 50:
                epsilon = max(epsilon_end, epsilon * 0.95)  # Faster decay initially
            else:
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # Track rewards
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-20:])  # Average over fewer episodes
            
            if self.verbose:
                logger.info(f"Episode {episode}/{num_episodes} | Reward: {episode_reward:.2f} | " +
                          f"Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.4f}")
            
            # Evaluate agent more frequently
            if episode % eval_freq == 0 or episode == num_episodes - 1:
                eval_reward = self.evaluate_agent(num_episodes=3)  # Fewer evaluation episodes
                logger.info(f"Evaluation at episode {episode}: Average Reward = {eval_reward:.2f}")
                
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self.agent.save_model('best_trading_agent.keras')
                    logger.info(f"Saved new best model with reward {best_reward:.2f}")
        
        # Save final model
        self.agent.save_model('final_trading_agent.keras')
        
        # Save scalers for later use
        self.data_processor.save_scalers(self.model_dir)
        
        logger.info(f"Trading agent training completed. Best evaluation reward: {best_reward:.2f}")
        
        # Load the best model for future use
        self.agent.load_model('best_trading_agent.keras')
        
        return best_reward
    
    def evaluate_agent(self, num_episodes: int = 5) -> float:
        """Evaluate the agent's performance."""
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
        self.price_predictor.load_model()
        self.agent.load_model('best_trading_agent.keras')
        self.data_processor.load_scalers(self.model_dir)
        logger.info("All models loaded successfully")
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run a backtest of the trading strategy."""
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
            "max_drawdown": performance_metrics['max_drawdown'],
            "total_trades": performance_metrics['num_trades'],
            "buyhold_value": buyhold_value,
            "vs_buyhold_pct": ((final_portfolio / buyhold_value) - 1) * 100,
            "trade_history": self.env.trade_history
        }
        
        return results


#-------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------

def main():
    """Main function to run the simplified trading system."""
    try:
        logger.info("Initializing CryptoTradingSystemLite...")
        
        # Create trading system
        trading_system = CryptoTradingSystemLite(
            ticker='TSLA',
            start_date='2024-01-01',
            end_date='2025-03-11',
            seq_length=30,  # Shorter sequence length
            initial_cash=10000,
            action_space_type='discrete',
            verbose=True
        )
        
        # Train price prediction model
        trading_system.train_predictive_model(epochs=20)
        
        # Train trading agent
        trading_system.train_agent(
            num_episodes=200,  # 200 episodes as requested
            batch_size=32,
            gamma=0.95,
            target_update_freq=5,
            save_freq=25
        )
        
        # Run backtest
        results = trading_system.run_backtest()
        
        # Print backtest results
        logger.info("\nBacktest Results:")
        logger.info(f"  Buy actions: {results['buy_count']}")
        logger.info(f"  Hold actions: {results['hold_count']}")
        logger.info(f"  Sell actions: {results['sell_count']}")
        logger.info(f"  Total trades: {results['total_trades']}")
        logger.info(f"  Initial portfolio: ${results['initial_value']:.2f}")
        logger.info(f"  Final portfolio: ${results['final_value']:.2f}")
        logger.info(f"  Performance: {results['gain_pct']:.2f}%")
        logger.info(f"  Maximum drawdown: {results['max_drawdown']:.2f}")
        logger.info(f"  Buy and hold value: ${results['buyhold_value']:.2f}")
        logger.info(f"  vs Buy-hold: {results['vs_buyhold_pct']:.2f}%")
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)


if __name__ == "__main__":
    main()