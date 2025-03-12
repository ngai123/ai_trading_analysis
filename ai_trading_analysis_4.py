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

# ---------------------- Prioritized Replay Buffer ----------------------
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for more efficient learning."""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.capacity = capacity
        self.alpha = alpha  # Degree of prioritization
        self.beta = beta  # Importance sampling correction factor
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # Small value to avoid zero priority
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch of experiences based on their priorities."""
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
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + self.epsilon

# ---------------------- Advanced Trading Environment ----------------------
class AdvancedTradingEnvironment:
    def __init__(self, trading_system, data, initial_cash=10000, max_position=0.5, 
                transaction_fee=0.001, slippage_model='proportional', 
                risk_management=True, use_market_hours=True, realistic_liquidity=True):
        """
        Advanced simulated trading environment for Bitcoin trading with more realistic features.
        
        Args:
            trading_system (AITradingSystem): Instance of the trading system
            data (np.ndarray): Scaled feature data
            initial_cash (float): Starting cash balance
            max_position (float): Maximum percentage of portfolio to hold in Bitcoin
            transaction_fee (float): Transaction fee as a percentage
            slippage_model (str): Model for simulating slippage ('proportional', 'fixed', 'market_impact')
            risk_management (bool): Whether to apply risk management rules
            use_market_hours (bool): Whether to consider market hours (24/7 for crypto)
            realistic_liquidity (bool): Whether to consider realistic liquidity constraints
        """
        self.trading_system = trading_system
        self.data = data
        self.raw_data = trading_system.raw_data
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.transaction_fee = transaction_fee
        self.slippage_model = slippage_model
        self.risk_management = risk_management
        self.use_market_hours = use_market_hours
        self.realistic_liquidity = realistic_liquidity
        
        # Risk management parameters
        self.stop_loss_pct = 0.05  # 5% stop loss from entry price
        self.take_profit_pct = 0.10  # 10% take profit from entry price
        self.max_drawdown_pct = 0.20  # Maximum drawdown allowed before reducing position
        self.max_trade_size_pct = 0.10  # Maximum size of any single trade as % of portfolio
        
        # Action space parameters (more granular actions)
        self.discrete_actions = 11  # -100%, -80%, -60%, -40%, -20%, 0%, 20%, 40%, 60%, 80%, 100%
        
        # Continuous action space - need to ensure compatibility with AITradingSystem
        self.use_continuous_actions = getattr(trading_system, 'use_continuous_action_space', False)
        
        # Trading frequency limits (per time period)
        self.max_trades_per_day = 5
        self.trade_cooldown = 3  # Minimum steps between trades
        
        # Market parameters
        self.liquidity_factor = 0.5  # How much liquidity is available (0-1)
        
        # Performance tracking
        self.trade_history = []
        self.portfolio_values = []
        self.trades_today = 0
        self.last_trade_step = 0
        self.entry_prices = {}  # Track entry prices for stop loss/take profit
        
        # Initialize environment
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.cash = self.initial_cash
        self.bitcoin_held = 0
        self.current_step = self.trading_system.seq_length
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
    
    def _get_state(self):
        """Get the current state for the DQN model."""
        # Get price history
        price_history = self.data[self.current_step - self.trading_system.seq_length:self.current_step]
        
        # Get current price
        current_price_scaled = self.data[self.current_step, 0]  # Scaled 'Close'
        current_price = self.trading_system.price_scaler.inverse_transform([[current_price_scaled]])[0, 0]
        
        # Calculate position metrics
        portfolio_value = self.cash + self.bitcoin_held * current_price
        bitcoin_value = self.bitcoin_held * current_price
        bitcoin_position_pct = bitcoin_value / portfolio_value if portfolio_value > 0 else 0
        unrealized_pnl = portfolio_value - self.initial_cash
        pnl_pct = unrealized_pnl / self.initial_cash if self.initial_cash > 0 else 0
        
        # Calculate technical indicators
        predicted_price = self.trading_system.predict_price(price_history)
        price_change_pct = (current_price / self.trading_system.price_scaler.inverse_transform(
            [[self.data[self.current_step-1, 0]]])[0, 0]) - 1 if self.current_step > 0 else 0
        
        # Market state from additional features
        volatility = self.data[self.current_step, 8] if self.data.shape[1] > 8 else 0  # Volatility feature
        volume = self.data[self.current_step, 1] if self.data.shape[1] > 1 else 0      # Volume feature
        sentiment = 0.0  # Default sentiment score (can be enhanced later)
        
        # Trading restrictions
        days_since_last_trade = (self.current_step - self.last_trade_step) / 24  # Approx. for crypto
        trade_allowed = self.trades_today < self.max_trades_per_day and (self.current_step - self.last_trade_step) >= self.trade_cooldown
        
        # Concatenate all state information
        state = np.concatenate([
            price_history.flatten(),
            [
                current_price,
                self.cash,
                self.bitcoin_held,
                bitcoin_position_pct,
                pnl_pct,
                predicted_price,
                price_change_pct,
                volatility,
                volume,
                sentiment,
                days_since_last_trade,
                float(trade_allowed)
            ]
        ])
        
        return state
    
    def calculate_slippage(self, action_size, current_price, volume):
        """Calculate slippage based on action size and market conditions."""
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
    
    def check_risk_management(self, action, bitcoin_value_pct, current_price):
        """Apply risk management rules to potentially modify action."""
        if not self.risk_management:
            return action
        
        # Check for stop loss / take profit for existing position
        if self.bitcoin_held > 0 and len(self.entry_prices) > 0:
            # Calculate average entry price
            avg_entry = sum(price * qty for price, qty in self.entry_prices.items()) / sum(qty for _, qty in self.entry_prices.items())
            pnl_pct = (current_price / avg_entry) - 1
            
            # Stop loss hit
            if pnl_pct < -self.stop_loss_pct:
                return 0  # Sell everything
            
            # Take profit hit
            if pnl_pct > self.take_profit_pct:
                return 2  # Take partial profits
        
        # Check for maximum drawdown
        max_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0
        if max_drawdown > self.max_drawdown_pct:
            # Reduce position size when drawdown is too large
            if action > 5:  # Buy actions
                return 5  # Hold instead
            if self.bitcoin_held > 0:
                return 3  # Reduce position slightly
        
        # Check for maximum position size
        if bitcoin_value_pct > self.max_position and action > 5:
            return 5  # Hold instead of buying more
        
        # Limit trade size for any single trade
        if abs(action - 5) / 5 > self.max_trade_size_pct:
            # Scale back the action to match maximum trade size
            direction = 1 if action > 5 else -1
            return 5 + direction * int(5 * self.max_trade_size_pct)
        
        return action
    
    def step(self, action):
        """
        Execute a trading action and return the next state, reward, and done flag.
        
        In continuous action space, action is a value between -1 and 1:
        - Negative values: sell proportion of holdings (e.g., -0.5 means sell 50% of holdings)
        - Zero: hold current position
        - Positive values: buy using proportion of cash (e.g., 0.5 means use 50% of cash to buy)
        
        In discrete action space, action is an integer between 0 and 10:
        - 0: Sell 100% of holdings
        - 1: Sell 80% of holdings
        - ...
        - 5: Hold (no action)
        - ...
        - 9: Buy with 80% of available cash
        - 10: Buy with 100% of available cash
        
        Original discrete actions (0=sell, 1=hold, 2=buy) are mapped to the enhanced space.
        """
        # Get current price
        current_price_scaled = self.data[self.current_step, 0]  # Scaled 'Close'
        current_price = self.trading_system.price_scaler.inverse_transform([[current_price_scaled]])[0, 0]
        
        # Get current volume for slippage calculation
        volume_scaled = self.data[self.current_step, 1] if self.data.shape[1] > 1 else 0.5
        
        # Calculate current portfolio state
        bitcoin_value = self.bitcoin_held * current_price
        portfolio_value = self.cash + bitcoin_value
        bitcoin_value_pct = bitcoin_value / portfolio_value if portfolio_value > 0 else 0
        
        # Process action depending on action space type
        if self.use_continuous_actions:
            # Continuous action space
            action_value = float(action)  # Should be between -1 and 1
            
            if action_value < -0.05:  # Sell
                sell_percentage = min(1.0, abs(action_value))
                sell_amount = self.bitcoin_held * sell_percentage
                
                # Calculate slippage for sell
                slippage_pct = self.calculate_slippage(sell_percentage, current_price, volume_scaled)
                effective_price = current_price * (1 - slippage_pct - self.transaction_fee)
                
                self.cash += sell_amount * effective_price
                self.bitcoin_held -= sell_amount
                
                # Record trade
                if sell_amount > 0:
                    self.trade_history.append({
                        'step': self.current_step,
                        'type': 'sell',
                        'amount': sell_amount,
                        'price': current_price,
                        'effective_price': effective_price,
                        'slippage': slippage_pct,
                        'fee': self.transaction_fee * sell_amount * current_price
                    })
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
                potential_bitcoin_value = (self.bitcoin_held + buy_amount) * current_price
                potential_position_pct = potential_bitcoin_value / (self.cash - cash_to_spend + potential_bitcoin_value)
                
                if potential_position_pct <= self.max_position:
                    self.bitcoin_held += buy_amount
                    self.cash -= cash_to_spend
                    
                    # Record entry price for this purchase
                    self.entry_prices[effective_price] = buy_amount
                    
                    # Record trade
                    self.trade_history.append({
                        'step': self.current_step,
                        'type': 'buy',
                        'amount': buy_amount,
                        'price': current_price,
                        'effective_price': effective_price,
                        'slippage': slippage_pct,
                        'fee': self.transaction_fee * cash_to_spend
                    })
                    self.last_trade_step = self.current_step
                    self.trades_today += 1
        
        else:
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
            action = self.check_risk_management(action, bitcoin_value_pct, current_price)
            
            # Map discrete action (0-10) to percentage (-100% to +100%)
            action_percentage = (action - 5) / 5  # Converts 0->-100%, 5->0%, 10->100%
            
            if action_percentage < -0.05:  # Sell
                sell_percentage = min(1.0, abs(action_percentage))
                sell_amount = self.bitcoin_held * sell_percentage
                
                # Calculate slippage for sell
                slippage_pct = self.calculate_slippage(sell_percentage, current_price, volume_scaled)
                effective_price = current_price * (1 - slippage_pct - self.transaction_fee)
                
                self.cash += sell_amount * effective_price
                self.bitcoin_held -= sell_amount
                
                # Record trade
                if sell_amount > 0:
                    self.trade_history.append({
                        'step': self.current_step,
                        'type': 'sell',
                        'amount': sell_amount,
                        'price': current_price,
                        'effective_price': effective_price,
                        'slippage': slippage_pct,
                        'fee': self.transaction_fee * sell_amount * current_price,
                        'original_action': original_action,
                        'modified_action': action
                    })
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
                potential_bitcoin_value = (self.bitcoin_held + buy_amount) * current_price
                potential_position_pct = potential_bitcoin_value / (self.cash - cash_to_spend + potential_bitcoin_value)
                
                if potential_position_pct <= self.max_position:
                    self.bitcoin_held += buy_amount
                    self.cash -= cash_to_spend
                    
                    # Record entry price for this purchase
                    self.entry_prices[effective_price] = buy_amount
                    
                    # Record trade
                    self.trade_history.append({
                        'step': self.current_step,
                        'type': 'buy',
                        'amount': buy_amount,
                        'price': current_price,
                        'effective_price': effective_price,
                        'slippage': slippage_pct,
                        'fee': self.transaction_fee * cash_to_spend,
                        'original_action': original_action,
                        'modified_action': action
                    })
                    self.last_trade_step = self.current_step
                    self.trades_today += 1
        
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
            # Get new price for next step
            next_price_scaled = self.data[self.current_step, 0]
            next_price = self.trading_system.price_scaler.inverse_transform([[next_price_scaled]])[0, 0]
            
            # Calculate new portfolio value
            new_portfolio_value = self.cash + self.bitcoin_held * next_price
            self.portfolio_values.append(new_portfolio_value)
            
            # Update maximum portfolio value for drawdown calculation
            if new_portfolio_value > self.max_portfolio_value:
                self.max_portfolio_value = new_portfolio_value
            
            # Calculate reward: combination of return and risk-adjusted metrics
            simple_return = new_portfolio_value - self.portfolio_value
            
            # Sharpe-like ratio component (reward / risk)
            returns_history = np.diff(self.portfolio_values[-20:]) if len(self.portfolio_values) > 20 else np.diff(self.portfolio_values)
            stddev = np.std(returns_history) if len(returns_history) > 0 else 1
            sharpe_component = simple_return / (stddev + 1e-9)  # Avoid division by zero
            
            # Drawdown penalty
            drawdown = (self.max_portfolio_value - new_portfolio_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0
            drawdown_penalty = -100 * drawdown if drawdown > self.max_drawdown_pct else 0
            
            # Inactivity penalty - small penalty for not trading for long periods
            inactivity_penalty = -0.1 * (self.current_step - self.last_trade_step) / 100 if self.current_step - self.last_trade_step > 100 else 0
            
            # Combined reward
            reward = simple_return + 0.1 * sharpe_component + drawdown_penalty + inactivity_penalty
            
            self.portfolio_value = new_portfolio_value
            done = False
            next_state = self._get_state()
        
        return next_state, reward, done
    
    def render(self, mode='human'):
        """Visualize the current state of the environment."""
        if mode == 'human':
            # Current portfolio state
            current_price_scaled = self.data[self.current_step - 1, 0]
            current_price = self.trading_system.price_scaler.inverse_transform([[current_price_scaled]])[0, 0]
            bitcoin_value = self.bitcoin_held * current_price
            portfolio_value = self.cash + bitcoin_value
            
            print(f"\n=== Step {self.current_step} ===")
            print(f"Cash: ${self.cash:.2f}")
            print(f"Bitcoin: {self.bitcoin_held:.8f} BTC (${bitcoin_value:.2f})")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(f"Initial Cash: ${self.initial_cash:.2f}")
            print(f"Return: {((portfolio_value / self.initial_cash) - 1) * 100:.2f}%")
            print(f"Max Drawdown: {((self.max_portfolio_value - portfolio_value) / self.max_portfolio_value) * 100:.2f}%")
            
            # Recent trades
            if len(self.trade_history) > 0:
                recent_trades = self.trade_history[-3:]
                print("\nRecent Trades:")
                for trade in recent_trades:
                    trade_type = trade['type'].upper()
                    print(f"{trade_type} {trade['amount']:.8f} BTC at ${trade['price']:.2f} (effective: ${trade['effective_price']:.2f})")
        
        return None
    
    def get_performance_metrics(self):
        """Calculate and return key performance metrics."""
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
        sharpe_ratio = np.mean(daily_returns_array) / (np.std(daily_returns_array) + 1e-9) * np.sqrt(365)
        
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
            if trade['type'] == 'buy':
                positions[trade['step']] = {
                    'price': trade['effective_price'],
                    'amount': trade['amount']
                }
            elif trade['type'] == 'sell':
                # Find oldest open position
                sell_amount_remaining = trade['amount']
                sell_price = trade['effective_price']
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

# ---------------------- AI Trading System ----------------------
class AITradingSystem:
    def __init__(self, ticker='BTC-USD', start_date='2020-01-01', end_date='2024-03-11', seq_length=60, 
                 verbose=True, use_continuous_action_space=False):
        self.ticker = ticker
        self.seq_length = seq_length
        self.start_date = start_date
        self.end_date = end_date
        self.verbose = verbose
        self.use_continuous_action_space = use_continuous_action_space
        
        # Action space configuration
        self.action_size = 1 if use_continuous_action_space else 3  # 1 continuous value or 3 discrete actions
        
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
        
        # Initialize the advanced trading environment
        self.env = AdvancedTradingEnvironment(
            self, self.scaled_features, 
            initial_cash=10000,
            max_position=0.5,
            transaction_fee=TRANSACTION_FEE,
            slippage_model='proportional',
            risk_management=True
        )

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
        """Create enhanced predictive and decision-making models."""
        # Use GPU if available
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("Using GPU for training")
        
        self.predictive_model = self.create_predictive_model_improved()
        self.lstm_model = self.create_lstm_model_improved()
        self.dqn_model = self.create_dqn_model_improved()

    def create_predictive_model_improved(self):
        """Enhanced CNN-based model for price prediction."""
        input_layer = keras.layers.Input(shape=(self.seq_length, self.X.shape[2]))
        conv1 = keras.layers.Conv1D(64, 2, activation='relu', padding='same')(input_layer)
        bn1 = keras.layers.BatchNormalization()(conv1)
        conv2 = keras.layers.Conv1D(64, 3, activation='relu', padding='same')(input_layer)
        bn2 = keras.layers.BatchNormalization()(conv2)
        conv3 = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(input_layer)
        bn3 = keras.layers.BatchNormalization()(conv3)
        merged = keras.layers.Concatenate()([bn1, bn2, bn3])
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
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.Huber())
        
        # Callbacks and training
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        checkpoint_path = os.path.join(self.model_dir, 'best_cnn_model.keras')
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss')
        model.fit(self.X, self.y, epochs=10, batch_size=64, validation_split=0.2, 
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
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.Huber())
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.model_dir, 'best_lstm_model.keras'),
                                            save_best_only=True, monitor='val_loss'),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        model.fit(self.X, self.y, epochs=10, batch_size=64, validation_split=0.2, 
                  callbacks=callbacks, verbose=1 if self.verbose else 0)
        model = keras.models.load_model(os.path.join(self.model_dir, 'best_lstm_model.keras'))
        return model

    def create_dqn_model_improved(self):
        """Enhanced DQN model for trading decisions with support for continuous or discrete action spaces."""
        # For the Advanced Trading Environment, the state is more complex
        # Base input dimension on flattened sequence plus extra features from _get_state
        # Default extra features: current_price, cash, bitcoin_held, bitcoin_position_pct, pnl_pct,
        # predicted_price, price_change_pct, volatility, volume, sentiment,
        # days_since_last_trade, float(trade_allowed)
        extra_features = 12  
        input_dim = self.seq_length * self.X.shape[2] + extra_features
        
        input_layer = keras.layers.Input(shape=(input_dim,))
        norm = keras.layers.BatchNormalization()(input_layer)
        
        branch1 = keras.layers.Dense(256, activation='relu')(norm)
        branch1 = keras.layers.Dropout(0.3)(branch1)
        branch1 = keras.layers.Dense(128, activation='relu')(branch1)
        
        branch2 = keras.layers.Dense(64, activation='relu')(norm)
        
        merge = keras.layers.Concatenate()([branch1, branch2])
        dense = keras.layers.Dense(128, activation='relu')(merge)
        dense = keras.layers.Dropout(0.3)(dense)
        
        # Output depends on action space type
        if self.use_continuous_action_space:
            # For continuous action space, output a single value between -1 and 1
            output = keras.layers.Dense(1, activation='tanh')(dense)
        else:
            # For discrete action space, output Q-values for each action
            output = keras.layers.Dense(self.action_size, activation='linear')(dense)
        
        model = keras.Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=keras.losses.Huber())
        return model

    def predict_price(self, input_sequence):
        """Predict future price using an ensemble of CNN and LSTM models."""
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

    # ---------------------- Improved DQN Training using Prioritized Replay ----------------------
    def train_dqn(self, num_episodes=1000, batch_size=64, gamma=0.99, target_update_freq=10, 
                  save_freq=50, eval_freq=20, learning_starts=1000, double_dqn=True, dueling_dqn=True):
        """
        Train the DQN model using a prioritized replay buffer.
        """
        env = self.env
        buffer_size = 100000
        replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        
        # Epsilon-greedy exploration parameters
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        epsilon = epsilon_start
        
        # Create target network by cloning the main model
        # This ensures weight compatibility between models
        target_model = tf.keras.models.clone_model(self.dqn_model)
        target_model.set_weights(self.dqn_model.get_weights())
        
        total_steps = 0
        best_reward = -float('inf')
        
        # Learning rate scheduler and optimizer
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.95
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        loss_fn = tf.keras.losses.Huber()
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Choose action
                if np.random.rand() < epsilon or total_steps < learning_starts:
                    if self.use_continuous_action_space:
                        action = np.random.uniform(-1, 1)
                    else:
                        action = np.random.choice(self.action_size)
                else:
                    state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
                    if self.use_continuous_action_space:
                        action = self.dqn_model(state_tensor)[0, 0].numpy()
                    else:
                        q_values = self.dqn_model(state_tensor)
                        action = np.argmax(q_values[0])
                
                next_state, reward, done = env.step(action)
                episode_reward += reward
                
                if next_state is not None:
                    replay_buffer.add(state, action, reward, next_state, done)
                    
                    if total_steps >= learning_starts and total_steps % 4 == 0:
                        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)
                        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
                        
                        with tf.GradientTape() as tape:
                            if self.use_continuous_action_space:
                                # For continuous action space
                                current_q = self.dqn_model(states_tensor)
                                current_q_values = current_q[:, 0]
                                
                                if double_dqn:
                                    next_q_main = self.dqn_model(next_states)
                                    next_actions = tf.clip_by_value(next_q_main, -1, 1)
                                    next_q_target = target_model(next_states)
                                    next_q_values = next_q_target[:, 0]
                                else:
                                    next_q_values = target_model(next_states)[:, 0]
                            else:
                                # For discrete action space
                                current_q = self.dqn_model(states_tensor)
                                current_q_values = tf.reduce_sum(current_q * tf.one_hot(tf.cast(actions, tf.int32), self.action_size), axis=1)
                                
                                if double_dqn:
                                    next_q_main = self.dqn_model(next_states)
                                    next_actions = tf.argmax(next_q_main, axis=1)
                                    next_q_target = target_model(next_states)
                                    next_q_values = tf.reduce_sum(next_q_target * tf.one_hot(next_actions, self.action_size), axis=1)
                                else:
                                    next_q_values = tf.reduce_max(target_model(next_states), axis=1)
                            
                            target_q_values = rewards + gamma * next_q_values * (1 - dones)
                            td_errors = tf.abs(current_q_values - target_q_values).numpy()
                            
                            loss = tf.reduce_mean(loss_fn(target_q_values, current_q_values) * weights)
                        
                        replay_buffer.update_priorities(indices, td_errors)
                        grads = tape.gradient(loss, self.dqn_model.trainable_variables)
                        grads, _ = tf.clip_by_global_norm(grads, 10.0)
                        optimizer.apply_gradients(zip(grads, self.dqn_model.trainable_variables))
                
                    state = next_state
                    total_steps += 1
                
                if total_steps % (target_update_freq * batch_size) == 0:
                    target_model.set_weights(self.dqn_model.get_weights())
            
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            print(f"Episode {episode}/{num_episodes} | Total Reward: {episode_reward:.2f} | Epsilon: {epsilon:.4f}")
            
            if episode % eval_freq == 0:
                eval_reward = self.evaluate_agent(num_episodes=5)
                print(f"Evaluation at episode {episode}: Average Reward = {eval_reward:.2f}")
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self.dqn_model.save(os.path.join(self.model_dir, 'best_dqn_model.keras'))
                    print(f"Saved new best model with reward {best_reward:.2f}")
            
            if episode % save_freq == 0:
                self.dqn_model.save(os.path.join(self.model_dir, f'dqn_model_episode_{episode}.keras'))
        
        self.dqn_model.save(os.path.join(self.model_dir, 'final_dqn_model.keras'))
        print(f"DQN training completed. Best evaluation reward: {best_reward:.2f}")
        self.dqn_model = keras.models.load_model(os.path.join(self.model_dir, 'best_dqn_model.keras'))
        return best_reward

    def create_dueling_dqn_model(self):
        """Create a dueling DQN architecture for better value estimation."""
        # Adjust input dimension to match the enhanced state representation
        extra_features = 12  # Same as in create_dqn_model_improved
        input_dim = self.seq_length * self.X.shape[2] + extra_features
        
        input_layer = keras.layers.Input(shape=(input_dim,))
        features = keras.layers.Dense(256, activation='relu')(input_layer)
        features = keras.layers.Dropout(0.3)(features)
        features = keras.layers.Dense(128, activation='relu')(features)
        
        # Value stream
        value_stream = keras.layers.Dense(64, activation='relu')(features)
        value = keras.layers.Dense(1)(value_stream)
        
        if self.use_continuous_action_space:
            # For continuous action, we output a single advantage that's added to the value
            advantage_stream = keras.layers.Dense(64, activation='relu')(features)
            advantage = keras.layers.Dense(1)(advantage_stream)
            
            # Output a single continuous value between -1 and 1
            raw_output = keras.layers.Add()([value, advantage])
            output = keras.layers.Lambda(lambda x: tf.tanh(x))(raw_output)
        else:
            # Advantage stream for discrete actions
            advantage_stream = keras.layers.Dense(64, activation='relu')(features)
            advantage = keras.layers.Dense(self.action_size)(advantage_stream)
            
            # Combine value and advantage
            advantage_mean = keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
            output = keras.layers.Add()([
                value,
                keras.layers.Subtract()([advantage, advantage_mean])
            ])
        
        model = keras.Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='huber_loss')
        return model

    def evaluate_agent(self, num_episodes=10):
        """Evaluate the agent's performance without exploration."""
        env = self.env
        total_rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
                
                if self.use_continuous_action_space:
                    action = self.dqn_model(state_tensor)[0, 0].numpy()
                else:
                    q_values = self.dqn_model(state_tensor)
                    action = np.argmax(q_values[0])
                
                next_state, reward, done = env.step(action)
                episode_reward += reward
                if next_state is not None:
                    state = next_state
                else:
                    break
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)

# ---------------------- Trading Simulation and Main ----------------------
def simulate_trading(trading_system):
    """Simulate trading with the trained DQN model."""
    print("Running trading simulation...")
    state = trading_system.env.reset()
    done = False
    buy_count = sell_count = hold_count = 0
    initial_portfolio = trading_system.env.portfolio_value
    
    while not done:
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        
        if trading_system.use_continuous_action_space:
            action_value = trading_system.dqn_model(state_tensor)[0, 0].numpy()
            if action_value < -0.05:
                sell_count += 1
            elif action_value > 0.05:
                buy_count += 1
            else:
                hold_count += 1
        else:
            q_values = trading_system.dqn_model(state_tensor)
            action = np.argmax(q_values[0])
            if action == 0:
                sell_count += 1
            elif action == 1:
                hold_count += 1
            else:
                buy_count += 1
                
        state, reward, done = trading_system.env.step(action if not trading_system.use_continuous_action_space else action_value)
    
    final_portfolio = trading_system.env.portfolio_value
    performance_metrics = trading_system.env.get_performance_metrics()
    
    return {
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "initial_value": initial_portfolio,
        "final_value": final_portfolio,
        "gain_pct": ((final_portfolio / initial_portfolio) - 1) * 100,
        "sharpe_ratio": performance_metrics['sharpe_ratio'],
        "max_drawdown": performance_metrics['max_drawdown'],
        "win_rate": performance_metrics['win_rate'],
        "total_trades": performance_metrics['num_trades']
    }

def main():
    try:
        print("Initializing Enhanced AI Trading System...")
        # You can switch between discrete and continuous action spaces
        use_continuous = False  # Set to True for continuous action space
        
        trading_system = AITradingSystem(
            ticker='BTC-USD', 
            start_date='2020-01-01', 
            end_date='2024-03-11',
            seq_length=60,
            verbose=True,
            use_continuous_action_space=use_continuous
        )
        
        # Train the DQN model (adjust num_episodes for improved results)
        best_eval_reward = trading_system.train_dqn(
            num_episodes=1,  # Increase for better results (e.g., 100, 500)
            double_dqn=True,
            dueling_dqn=True
        )
        
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
        
        # Simulate trading with the trained model
        results = simulate_trading(trading_system)
        print("\nTrading Simulation Results:")
        print(f"  Buy actions: {results['buy_count']}")
        print(f"  Hold actions: {results['hold_count']}")
        print(f"  Sell actions: {results['sell_count']}")
        print(f"  Total trades: {results['total_trades']}")
        print(f"  Win rate: {results['win_rate']:.2f}")
        print(f"  Initial portfolio: ${results['initial_value']:.2f}")
        print(f"  Final portfolio: ${results['final_value']:.2f}")
        print(f"  Performance: {results['gain_pct']:.2f}%")
        print(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Maximum drawdown: {results['max_drawdown']:.2f}")
        
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