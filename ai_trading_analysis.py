import numpy as np
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
            trading_system (AITradingSystem): Instance of the trading system
            data (np.ndarray): Scaled feature data
            initial_cash (float): Starting cash balance
            max_position (float): Maximum percentage of portfolio to hold in Bitcoin
        """
        self.trading_system = trading_system
        self.data = data
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        self.cash = self.initial_cash
        self.bitcoin_held = 0
        self.current_step = self.trading_system.seq_length
        self.portfolio_value = self.cash
        return self._get_state()

    def _get_state(self):
        """Get the current state for the DQN model."""
        price_history = self.data[self.current_step - self.trading_system.seq_length:self.current_step]
        sentiment_score = np.random.uniform(-1, 1)  # Simulate sentiment (replace with real data)
        predicted_price = self.trading_system.predict_price(price_history)
        state = np.concatenate([price_history.flatten(), [sentiment_score, predicted_price, self.cash, self.bitcoin_held]])
        return state

    def step(self, action):
        """Execute an action and return the next state, reward, and done flag."""
        current_price_scaled = self.data[self.current_step, 0]  # Scaled 'Close'
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
    def __init__(self, ticker='BTC-USD', start_date='2020-01-01', end_date='2024-03-11', seq_length=60):
        self.ticker = ticker
        self.seq_length = seq_length
        self.raw_data = yf.download(ticker, start=start_date, end=end_date)
        if len(self.raw_data) < self.seq_length + 1:
            raise ValueError(f"Insufficient data. Need at least {self.seq_length + 1} data points.")
        self.prepare_data()
        self.create_models()
        self.env = TradingEnvironment(self, self.scaled_features)

    def prepare_data(self):
        """Prepare and preprocess financial data."""
        self.feature_scaler = MinMaxScaler()
        features = self.raw_data[['Close', 'Volume', 'High', 'Low']].values
        self.scaled_features = self.feature_scaler.fit_transform(features)
        self.price_scaler = MinMaxScaler()
        self.scaled_prices = self.price_scaler.fit_transform(self.raw_data[['Close']].values)
        self.X, self.y = self.create_sequences(self.scaled_features, self.scaled_prices)

    def create_sequences(self, data, targets):
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(targets[i + self.seq_length, 0])
        return np.array(X), np.array(y)

    def create_models(self):
        """Create predictive and decision-making models."""
        self.predictive_model = self.create_predictive_model()
        self.lstm_model = self.create_lstm_model()
        self.dqn_model = self.create_dqn_model()

    def create_predictive_model(self):
        """CNN-based model for price prediction."""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.seq_length, 4)),
            keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            keras.layers.GlobalMaxPooling1D(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(self.X, self.y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        return model

    def create_lstm_model(self):
        """LSTM model for time-series forecasting."""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.seq_length, 4)),
            keras.layers.LSTM(64, activation='tanh', return_sequences=True),
            keras.layers.LSTM(32, activation='tanh'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(self.X, self.y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        return model

    def create_dqn_model(self):
        """DQN model for trading decisions."""
        input_dim = self.seq_length * 4 + 4  # Flattened price history + sentiment + predicted_price + cash + bitcoin_held
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(3, activation='linear')  # Buy, hold, sell
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def predict_price(self, input_sequence):
        """Predict future price using ensemble of models."""
        try:
            if input_sequence.ndim == 2:
                input_sequence = input_sequence[np.newaxis, :, :]
            pred_nn = self.predictive_model.predict(input_sequence, verbose=0)[0, 0]
            pred_lstm = self.lstm_model.predict(input_sequence, verbose=0)[0, 0]
            combined_pred = (pred_nn + pred_lstm) / 2
            return self.price_scaler.inverse_transform([[combined_pred]])[0, 0]
        except Exception as e:
            print(f"Error in predict_price: {e}")
            return None

    def train_dqn(self, num_episodes=100, batch_size=32, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """Train the DQN model using reinforcement learning."""
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
                    action = np.random.choice(3)
                else:
                    q_values = self.dqn_model.predict(state[np.newaxis, :], verbose=0)
                    action = np.argmax(q_values[0])

                next_state, reward, done = self.env.step(action)
                total_reward += reward

                if next_state is not None:
                    replay_buffer.append((state, action, reward, next_state, done))
                    state = next_state

                if len(replay_buffer) > batch_size:
                    batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
                    states, actions, rewards, next_states, dones = zip(*[replay_buffer[i] for i in batch])
                    states = np.array(states)
                    next_states = np.array(next_states)
                    q_values_next = target_model.predict(next_states, verbose=0)
                    targets = rewards + gamma * np.max(q_values_next, axis=1) * (1 - np.array(dones))

                    q_values = self.dqn_model.predict(states, verbose=0)
                    for i, a in enumerate(actions):
                        q_values[i, a] = targets[i]

                    self.dqn_model.fit(states, q_values, epochs=1, verbose=0)

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if episode % 10 == 0:
                target_model.set_weights(self.dqn_model.get_weights())
                print(f"Episode {episode}, Total Reward: {total_reward}")

        print("DQN training completed.")

def main():
    try:
        trading_system = AITradingSystem()
        trading_system.train_dqn(num_episodes=50)  # Train for 50 episodes (adjust as needed)
        
        # Simulate trading with the trained DQN model
        state = trading_system.env.reset()
        done = False
        actions = []
        while not done:
            q_values = trading_system.dqn_model.predict(state[np.newaxis, :], verbose=0)
            action = np.argmax(q_values[0])
            actions.append(action)
            state, _, done = trading_system.env.step(action)
        
        print(f"Simulated trading actions: {actions}")
        print(f"Final portfolio value: ${trading_system.env.portfolio_value:.2f}")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()