import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Global transaction fee (0.1%)
TRANSACTION_FEE = 0.001

class TradingEnvironment:
    def __init__(self, trading_system, data, initial_cash=10000, max_position=0.5):
        """
        Simulated trading environment for DQN training.
        
        Args:
            trading_system (AITradingSystem): Instance of the trading system.
            data (np.ndarray): Scaled feature data.
            initial_cash (float): Starting cash balance.
            max_position (float): Maximum percentage of portfolio to hold in Bitcoin.
        """
        self.trading_system = trading_system
        self.data = data
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.reset()

    def reset(self):
        """Reset the environment to the initial state."""
        self.cash = self.initial_cash
        self.bitcoin_held = 0
        self.current_step = self.trading_system.seq_length
        self.portfolio_value = self.cash
        return self._get_state()

    def _get_state(self):
        """Get the current state for the DQN model."""
        price_history = self.data[self.current_step - self.trading_system.seq_length:self.current_step]
        # For simplicity, we use a fixed sentiment score.
        sentiment_score = 0.0  
        predicted_price = self.trading_system.predict_price(price_history)
        # State: flattened historical features + sentiment, predicted price, cash, bitcoin held.
        state = np.concatenate([price_history.flatten(), [sentiment_score, predicted_price, self.cash, self.bitcoin_held]])
        return state

    def step(self, action):
        """Execute an action and return the next state, reward, and done flag."""
        # Current price is based on the first feature (scaled 'Close')
        current_price_scaled = self.data[self.current_step, 0]  
        current_price = self.trading_system.price_scaler.inverse_transform([[current_price_scaled]])[0, 0]
        
        if action == 0:  # Sell 10% of held Bitcoin
            sell_amount = 0.1 * self.bitcoin_held
            self.cash += sell_amount * current_price * (1 - TRANSACTION_FEE)
            self.bitcoin_held -= sell_amount
        elif action == 2:  # Buy with 10% of cash, respecting max position
            cash_to_spend = 0.1 * self.cash
            buy_amount = cash_to_spend / current_price
            potential_position = (self.bitcoin_held + buy_amount) * current_price / self.portfolio_value
            if potential_position <= self.max_position:
                self.bitcoin_held += buy_amount
                self.cash -= cash_to_spend * (1 + TRANSACTION_FEE)
        
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
            reward = 0
            next_state = None
        else:
            next_price_scaled = self.data[self.current_step, 0]
            next_price = self.trading_system.price_scaler.inverse_transform([[next_price_scaled]])[0, 0]
            next_portfolio_value = self.cash + self.bitcoin_held * next_price
            reward = next_portfolio_value - self.portfolio_value
            self.portfolio_value = next_portfolio_value
            done = False
            next_state = self._get_state()
        return next_state, reward, done

class AITradingSystem:
    def __init__(self, ticker='BTC-USD', start_date='2020-01-01', end_date='2024-03-11', seq_length=60, verbose=True):
        self.ticker = ticker
        self.seq_length = seq_length
        self.start_date = start_date
        self.end_date = end_date
        self.verbose = verbose
        
        # Set additional attributes for improved model architectures
        self.use_continuous_action_space = False  # Discrete actions (sell, hold, buy)
        self.action_size = 3
        self.model_dir = "saved_models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Download and prepare the data
        print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
        self.raw_data = yf.download(ticker, start=start_date, end=end_date)
        if len(self.raw_data) < self.seq_length + 1:
            raise ValueError(f"Insufficient data. Need at least {self.seq_length + 1} data points.")
        
        self.prepare_data()
        self.create_models()
        self.env = TradingEnvironment(self, self.scaled_features)

    def prepare_data(self):
        """Prepare and preprocess financial data with enhanced features."""
        self.processed_data = self.raw_data.copy()
        
        # Technical indicators: moving averages
        self.processed_data['MA_7'] = self.processed_data['Close'].rolling(window=7).mean()
        self.processed_data['MA_21'] = self.processed_data['Close'].rolling(window=21).mean()
        
        # RSI calculation
        delta = self.processed_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean().replace(0, 0.001)  
        rs = avg_gain / avg_loss
        self.processed_data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility feature
        self.processed_data['Log_Return'] = np.log(self.processed_data['Close'] / self.processed_data['Close'].shift(1))
        self.processed_data['Volatility_21'] = self.processed_data['Log_Return'].rolling(window=21).std()
        
        # Fill missing values
        self.processed_data = self.processed_data.ffill().fillna(0)
        
        # Scale selected features
        feature_columns = ['Close', 'Volume', 'High', 'Low', 'MA_7', 'MA_21', 'RSI', 'Log_Return', 'Volatility_21']
        self.feature_scaler = MinMaxScaler()
        self.scaled_features = self.feature_scaler.fit_transform(self.processed_data[feature_columns])
        
        # Scale prices for prediction targets
        self.price_scaler = MinMaxScaler()
        self.scaled_prices = self.price_scaler.fit_transform(self.processed_data[['Close']])
        
        # Create time series sequences
        self.X, self.y = self.create_sequences(self.scaled_features, self.scaled_prices)
        print(f"Data prepared: {len(self.processed_data)} rows, {len(feature_columns)} features")

    def create_sequences(self, data, targets):
        """Create sequences for time series prediction."""
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(targets[i + self.seq_length, 0])
        return np.array(X), np.array(y)

    def create_models(self):
        """Create enhanced predictive and decision-making models using improved architectures."""
        # Configure GPU if available
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("Using GPU for training")
        
        # Create improved models
        self.predictive_model = self.create_predictive_model_improved()
        self.lstm_model = self.create_lstm_model_improved()
        self.dqn_model = self.create_dqn_model_improved()

    def create_predictive_model_improved(self):
        """Enhanced CNN-based model for price prediction."""
        input_layer = keras.layers.Input(shape=(self.seq_length, self.X.shape[2]))
        # Parallel convolutional branches for multi-scale feature extraction
        conv1 = keras.layers.Conv1D(64, 2, activation='relu', padding='same')(input_layer)
        bn1 = keras.layers.BatchNormalization()(conv1)
        conv2 = keras.layers.Conv1D(64, 3, activation='relu', padding='same')(input_layer)
        bn2 = keras.layers.BatchNormalization()(conv2)
        conv3 = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(input_layer)
        bn3 = keras.layers.BatchNormalization()(conv3)
        merged = keras.layers.Concatenate()([bn1, bn2, bn3])
        # Attention mechanism to focus on important features
        attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(merged, merged)
        attention = keras.layers.LayerNormalization()(attention + merged)
        pooling = keras.layers.GlobalAveragePooling1D()(attention)
        dense1 = keras.layers.Dense(128, activation='relu')(pooling)
        dropout1 = keras.layers.Dropout(0.3)(dense1)
        dense2 = keras.layers.Dense(64, activation='relu')(dropout1)
        dropout2 = keras.layers.Dropout(0.3)(dense2)
        merge_dense = keras.layers.Concatenate()([pooling, dropout2])
        output = keras.layers.Dense(1, activation='linear')(merge_dense)
        model = keras.Model(inputs=input_layer, outputs=output)
        
        # Use an exponential decay learning rate
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.Huber())
        
        # Set up callbacks for early stopping and checkpointing
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        checkpoint_path = os.path.join(self.model_dir, 'best_cnn_model.keras')
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss')
        
        model.fit(self.X, self.y, epochs=100, batch_size=64, validation_split=0.2, 
                  callbacks=[early_stopping, model_checkpoint], verbose=1 if self.verbose else 0)
        model = keras.models.load_model(checkpoint_path)
        return model

    def create_lstm_model_improved(self):
        """Enhanced LSTM model for time-series forecasting."""
        input_layer = keras.layers.Input(shape=(self.seq_length, self.X.shape[2]))
        lstm1 = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(input_layer)
        lstm2 = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(lstm1)
        concat = keras.layers.Concatenate()([lstm1, lstm2])
        attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(concat, concat)
        attention = keras.layers.LayerNormalization()(attention + concat)
        pooling = keras.layers.GlobalAveragePooling1D()(attention)
        dense1 = keras.layers.Dense(64, activation='relu')(pooling)
        dropout1 = keras.layers.Dropout(0.3)(dense1)
        output = keras.layers.Dense(1, activation='linear')(dropout1)
        model = keras.Model(inputs=input_layer, outputs=output)
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.Huber())

        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.model_dir, 'best_lstm_model.keras'),
                                            save_best_only=True, monitor='val_loss'),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        model.fit(self.X, self.y, epochs=100, batch_size=64, validation_split=0.2, 
                  callbacks=callbacks, verbose=1 if self.verbose else 0)
        model = keras.models.load_model(os.path.join(self.model_dir, 'best_lstm_model.keras'))
        return model

    def create_dqn_model_improved(self):
        """Enhanced DQN model for trading decisions."""
        # Note: The environment state includes flattened price history plus 4 extra state variables.
        input_dim = self.seq_length * self.X.shape[2] + 4
        input_layer = keras.layers.Input(shape=(input_dim,))
        norm = keras.layers.BatchNormalization()(input_layer)
        
        # Branch 1: Deep layers for complex pattern recognition
        branch1 = keras.layers.Dense(256, activation='relu')(norm)
        branch1 = keras.layers.Dropout(0.3)(branch1)
        branch1 = keras.layers.Dense(128, activation='relu')(branch1)
        
        # Branch 2: Shallow layer for direct feature extraction
        branch2 = keras.layers.Dense(64, activation='relu')(norm)
        
        merge = keras.layers.Concatenate()([branch1, branch2])
        dense = keras.layers.Dense(128, activation='relu')(merge)
        dense = keras.layers.Dropout(0.3)(dense)
        
        # Discrete action space output: 0 = sell, 1 = hold, 2 = buy
        output = keras.layers.Dense(self.action_size, activation='linear')(dense)
        model = keras.Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=keras.losses.Huber())
        return model

    def predict_price(self, input_sequence):
        """Predict future price using an ensemble of the predictive (CNN) and LSTM models."""
        try:
            if input_sequence.ndim == 2:
                input_sequence = input_sequence[np.newaxis, :, :]
            pred_cnn = self.predictive_model.predict(input_sequence, verbose=0)[0, 0]
            pred_lstm = self.lstm_model.predict(input_sequence, verbose=0)[0, 0]
            combined_pred = (pred_cnn + pred_lstm) / 2
            return self.price_scaler.inverse_transform([[combined_pred]])[0, 0]
        except Exception as e:
            print(f"Error in predict_price: {e}")
            return None

    def train_dqn(self, num_episodes=100, batch_size=32, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """Train the DQN model using reinforcement learning."""
        print(f"Training DQN for {num_episodes} episodes...")
        replay_buffer = []
        epsilon = epsilon_start
        target_model = tf.keras.models.clone_model(self.dqn_model)
        target_model.set_weights(self.dqn_model.get_weights())

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                if np.random.rand() < epsilon:
                    action = np.random.choice(self.action_size)
                else:
                    q_values = self.dqn_model.predict(state[np.newaxis, :], verbose=0)
                    action = np.argmax(q_values[0])

                next_state, reward, done = self.env.step(action)
                total_reward += reward

                if next_state is not None:
                    replay_buffer.append((state, action, reward, next_state, done))
                    state = next_state

                if len(replay_buffer) > batch_size:
                    batch_indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                    batch = [replay_buffer[i] for i in batch_indices]
                    
                    states = np.array([item[0] for item in batch])
                    actions = np.array([item[1] for item in batch])
                    rewards = np.array([item[2] for item in batch])
                    next_states = np.array([item[3] for item in batch])
                    dones = np.array([item[4] for item in batch])
                    
                    q_values_next = target_model.predict(next_states, verbose=0)
                    targets = rewards + gamma * np.max(q_values_next, axis=1) * (1 - dones)
                    
                    q_values = self.dqn_model.predict(states, verbose=0)
                    for i, a in enumerate(actions):
                        q_values[i, a] = targets[i]
                    
                    self.dqn_model.fit(states, q_values, epochs=1, verbose=0)

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            if episode % 5 == 0 or episode == num_episodes - 1:
                target_model.set_weights(self.dqn_model.get_weights())
                print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

        print("DQN training completed.")

def simulate_trading(trading_system):
    """Simulate trading with the trained DQN model."""
    print("Running trading simulation...")
    state = trading_system.env.reset()
    done = False
    buy_count = sell_count = hold_count = 0
    initial_portfolio = trading_system.env.portfolio_value
    
    while not done:
        q_values = trading_system.dqn_model.predict(state[np.newaxis, :], verbose=0)
        action = np.argmax(q_values[0])
        if action == 0:
            sell_count += 1
        elif action == 1:
            hold_count += 1
        else:
            buy_count += 1
            
        state, reward, done = trading_system.env.step(action)
    
    final_portfolio = trading_system.env.portfolio_value
    return {
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "initial_value": initial_portfolio,
        "final_value": final_portfolio,
        "gain_pct": ((final_portfolio / initial_portfolio) - 1) * 100
    }

def main():
    try:
        print("Initializing AI Trading System...")
        trading_system = AITradingSystem()
        
        # Train the DQN model (adjust num_episodes for better results)
        trading_system.train_dqn(num_episodes=1)
        
        # Save the trained DQN model
        model_path = os.path.join(trading_system.model_dir, 'bitcoin_model.keras')
        trading_system.dqn_model.save(model_path)
        print(f"Model saved to {model_path}")

        # Save the price scaler as JSON
        scaler_path = os.path.join(trading_system.model_dir, 'scaler.json')
        scaler_data = {
            "min_": trading_system.price_scaler.min_.tolist(),
            "scale_": trading_system.price_scaler.scale_.tolist()
        }
        with open(scaler_path, 'w') as f:
            json.dump(scaler_data, f)
        print(f"Scaler saved to {scaler_path}")
        
        # Simulate trading
        results = simulate_trading(trading_system)
        print("\nTrading Simulation Results:")
        print(f"  Buy actions: {results['buy_count']}")
        print(f"  Hold actions: {results['hold_count']}")
        print(f"  Sell actions: {results['sell_count']}")
        print(f"  Initial portfolio: ${results['initial_value']:.2f}")
        print(f"  Final portfolio: ${results['final_value']:.2f}")
        print(f"  Performance: {results['gain_pct']:.2f}%")
        
        # Compare to a buy-and-hold strategy
        first_price = trading_system.price_scaler.inverse_transform([[trading_system.scaled_features[trading_system.seq_length, 0]]])[0, 0]
        last_price = trading_system.price_scaler.inverse_transform([[trading_system.scaled_features[-1, 0]]])[0, 0]
        initial_btc = 10000 / first_price
        buyhold_value = initial_btc * last_price
        print(f"  Buy and hold value: ${buyhold_value:.2f}")
        print(f"  vs Buy-hold: {(results['final_value'] / buyhold_value - 1) * 100:.2f}%")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
