# Stock-Price-Predictor
This project uses an LSTM-based neural network to predict future stock prices using historical market data and a sentiment factor that adjusts based on user-rated news sentiment.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock_price_predictor.git
cd stock_price_predictor
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run the predictor
```
python main.py
```

### 4. Run the backtest (Optional)

```
#Inside main.py
test_nvda_prediction(news_sentiment=7.0)
```

## How it works

- Uses the yfinance dependency to grab stock historical data
- Applies MinMax scaling
- Addes a synthetic sentiment news feature based on user's own opinions
- Trains an LSTM model to predict future prices
- Can compare predictions with actual market history

## Output
- Model training RMSE
- Line of plot of past, predicted, and actual future prices

## Author
Created by Alex Seummalavanh. Open to contributions and improvements