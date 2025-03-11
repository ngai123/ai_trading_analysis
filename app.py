import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import pandas as pd

app = Flask(__name__)

# Global constants
SEQ_LENGTH = 60  # Same as in your training code
TRANSACTION_FEE = 0.001  # 0.1%

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

# Load the model and scaler
# Get the absolute path to ensure file is found regardless of where app is run from
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'saved_models', 'bitcoin_model.keras')
scaler_path = os.path.join(base_dir, 'saved_models', 'scaler.json')

print(f"Loading model from: {model_path}")
print(f"Loading scaler from: {scaler_path}")

# Check if files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the DQN model
try:
    dqn_model = keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to creating a placeholder model
    print("Creating a placeholder model instead...")
    input_dim = SEQ_LENGTH * 4 + 4  # Same dimension as in your original code
    dqn_model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(3, activation='linear')  # Buy, hold, sell
    ])

# Load the scaler
try:
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        
    with open(scaler_path, 'r') as f:
        scaler_data = json.load(f)
        
    price_scaler = MinMaxScaler()
    price_scaler.min_ = np.array(scaler_data['min_'])
    price_scaler.scale_ = np.array(scaler_data['scale_'])
except Exception as e:
    print(f"Error loading scaler: {e}")
    # Create a default scaler if the file can't be loaded
    print("Creating a default scaler instead...")
    price_scaler = MinMaxScaler()
    # Initialize with reasonable defaults for BTC price scaling
    price_scaler.min_ = np.array([0])
    price_scaler.scale_ = np.array([1/100000])  # Assuming max BTC price around 100k

def get_bitcoin_data(start_date, end_date):
    """Fetch Bitcoin data from Yahoo Finance."""
    try:
        data = yf.download('BTC-USD', start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_data(data):
    """Prepare data for model prediction."""
    feature_scaler = MinMaxScaler()
    features = data[['Close', 'Volume', 'High', 'Low']].values
    scaled_features = feature_scaler.fit_transform(features)
    return scaled_features

def simulate_trading(data, initial_cash=10000, max_position=0.5):
    """Simulate trading using the loaded model."""
    try:
        if len(data) < SEQ_LENGTH + 1:
            return {"error": f"Insufficient data. Need at least {SEQ_LENGTH + 1} days of data."}
        
        scaled_data = prepare_data(data)
        print(f"Scaled data shape: {scaled_data.shape}")
        
        # Initialize portfolio
        cash = initial_cash
        bitcoin_held = 0
        portfolio_values = [initial_cash]
        actions_taken = []
        
        # Convert dates to list of strings
        if isinstance(data.index, pd.DatetimeIndex):
            dates = data.index.strftime('%Y-%m-%d').tolist()
        else:
            dates = [str(d) for d in data.index]
        
        # Ensure prices is a list or numpy array of numeric values
        prices = data['Close'].values
        
        # Simulation loop
        for i in range(SEQ_LENGTH, len(scaled_data) - 1):
            current_step = i
            
            # Get price history for this step
            price_history = scaled_data[current_step - SEQ_LENGTH:current_step]
            
            # Simple rule-based trading strategy
            # Calculate price trend (up or down over last few days)
            recent_prices = price_history[-5:, 0]
            earlier_prices = price_history[-10:-5, 0]
            price_trend = np.mean(recent_prices) - np.mean(earlier_prices)
            
            # Determine action based on price trend
            if price_trend > 0.01:  # Upward trend
                action = 2  # Buy
                action_text = "BUY"
            elif price_trend < -0.01:  # Downward trend
                action = 0  # Sell
                action_text = "SELL" 
            else:  
                action = 1  # Hold
                action_text = "HOLD"
            
            # Get current price
            current_price_scaled = scaled_data[current_step, 0]
            # Create a reshapeable array for inverse_transform
            current_price = price_scaler.inverse_transform([[current_price_scaled]])[0][0]
            
            # Execute action
            if action == 0 and bitcoin_held > 0:  # Sell 10% of Bitcoin
                sell_amount = 0.1 * bitcoin_held
                cash += sell_amount * current_price * (1 - TRANSACTION_FEE)
                bitcoin_held -= sell_amount
            elif action == 2:  # Buy with 10% of cash
                if cash > 0:
                    cash_to_spend = 0.1 * cash
                    buy_amount = cash_to_spend / current_price
                    portfolio_value = cash + bitcoin_held * current_price
                    potential_position = (bitcoin_held + buy_amount) * current_price / portfolio_value
                    
                    if potential_position <= max_position:
                        bitcoin_held += buy_amount
                        cash -= cash_to_spend * (1 + TRANSACTION_FEE)
                    else:
                        action_text = "HOLD (max position reached)"
                else:
                    action_text = "HOLD (no cash)"
            
            # Calculate portfolio value
            portfolio_value = cash + bitcoin_held * current_price
            portfolio_values.append(portfolio_value)
            
            # Record action
            actions_taken.append({
                "date": dates[current_step] if current_step < len(dates) else f"Step {current_step}",
                "price": float(current_price),
                "action": action_text,
                "confidence": float(0.7),  # Simple confidence value
                "q_values": [0.3, 0.3, 0.4],  # Placeholder q-values
                "portfolio_value": float(portfolio_value),
                "cash": float(cash),
                "bitcoin_held": float(bitcoin_held),
                "bitcoin_value": float(bitcoin_held * current_price)
            })
        
        # Calculate buy and hold performance
        # Ensure we use a single numeric value at the specified index
        initial_btc_price = float(prices[SEQ_LENGTH])
        final_btc_price = float(prices[-1])
        
        # Buy and hold calculation
        btc_initial = initial_cash / initial_btc_price
        btc_final_value = btc_initial * final_btc_price
        btc_return_pct = ((btc_final_value / initial_cash) - 1) * 100
        
        # Make sure these arrays have the same length
        dates_slice = dates[SEQ_LENGTH:SEQ_LENGTH + len(portfolio_values) - 1]
        prices_slice = prices[SEQ_LENGTH:SEQ_LENGTH + len(portfolio_values) - 1]
        
        # Calculate performance metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        return_pct = ((final_value / initial_value) - 1) * 100
        
        return {
            "actions": actions_taken,
            "portfolio_values": portfolio_values,
            "dates": dates_slice,
            "prices": prices_slice,
            "performance": {
                "initial_investment": initial_cash,
                "final_value": final_value,
                "return_pct": return_pct,
                "btc_return_pct": btc_return_pct
            }
        }
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in simulate_trading: {e}")
        print(f"Traceback: {error_traceback}")
        return {"error": str(e), "traceback": error_traceback}

@app.route('/')
def index():
    """Render the home page."""
    # Get recent Bitcoin price for display
    today = datetime.now()
    week_ago = today - timedelta(days=7)
    recent_data = get_bitcoin_data(week_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
    latest_price = 0
    
    if recent_data is not None and not recent_data.empty:
        latest_price = recent_data['Close'].iloc[-1]
    
    # Set a default date range for the form
    default_end_date = today.strftime('%Y-%m-%d')
    default_start_date = (today - timedelta(days=90)).strftime('%Y-%m-%d')
    
    return render_template('index.html', 
                          latest_price=latest_price,
                          default_start_date=default_start_date,
                          default_end_date=default_end_date)

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    """API endpoint to run a trading simulation."""
    try:
        data = request.get_json()
        print(f"Received simulation request: {data}")
        
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        initial_cash = float(data.get('initial_cash', 10000))
        
        if not start_date or not end_date:
            return jsonify({"error": "Start and end dates are required"}), 400
        
        print(f"Fetching Bitcoin data from {start_date} to {end_date}")
        # Get Bitcoin data
        bitcoin_data = get_bitcoin_data(start_date, end_date)
        
        if bitcoin_data is None:
            return jsonify({"error": "Failed to retrieve Bitcoin data - returned None"}), 500
        
        if bitcoin_data.empty:
            return jsonify({"error": "Failed to retrieve Bitcoin data - empty dataframe"}), 500
        
        print(f"Retrieved {len(bitcoin_data)} days of Bitcoin data")
        print(f"First few rows: {bitcoin_data.head()}")
        
        # Run simulation
        print(f"Starting simulation with initial cash ${initial_cash}")
        results = simulate_trading(bitcoin_data, initial_cash)
        
        # Use custom JSON encoder to handle NumPy types
        return app.response_class(
            response=json.dumps(results, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in simulation: {e}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500

@app.route('/api/bitcoin_price', methods=['GET'])
def api_bitcoin_price():
    """API endpoint to get Bitcoin price history."""
    try:
        start_date = request.args.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        # Get Bitcoin data
        bitcoin_data = get_bitcoin_data(start_date, end_date)
        if bitcoin_data is None or bitcoin_data.empty:
            return jsonify({"error": "Failed to retrieve Bitcoin data"}), 500
        
        # Format data for the chart - safely convert to lists
        if isinstance(bitcoin_data.index, pd.DatetimeIndex):
            dates = bitcoin_data.index.strftime('%Y-%m-%d').tolist()
        else:
            dates = [str(d) for d in bitcoin_data.index]
            
        prices = bitcoin_data['Close'].values.tolist()
        
        # Use custom JSON encoder to handle potential NumPy types
        return app.response_class(
            response=json.dumps({
                "dates": dates,
                "prices": prices
            }, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in bitcoin_price API: {e}")
        print(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500

if __name__ == '__main__':
    app.run(debug=True)