import datetime
import numpy as np
from data.generator import StockDataGenerator
from models.lstm_model import build_model
from utils.helpers import predict_future_prices
import yfinance as yf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def main(news_sentiment):
    ticker = "NVDA"
    TIME_STEP = 60
    BATCH_SIZE = 30
    EPOCHS = 400
    PAST_DAYS = 60
    FUTURE_DAYS = 60
    MIN_REQUIRED_DAYS = TIME_STEP + 30

    now = datetime.datetime.now()
    test_start = now - datetime.timedelta(days=240 + PAST_DAYS + FUTURE_DAYS)
    test_middle = test_start + datetime.timedelta(days=PAST_DAYS + MIN_REQUIRED_DAYS)
    test_end = test_middle + datetime.timedelta(days=FUTURE_DAYS)

    past_gen = StockDataGenerator(test_start, test_middle, ticker, news_sentiment, BATCH_SIZE)
    model = build_model(input_shape=(TIME_STEP, past_gen.X.shape[2]))
    model.fit(past_gen, epochs=EPOCHS)

    predicted_10 = predict_future_prices(model, past_gen.last_sequence, 10, past_gen.scaler)
    predicted_5 = predict_future_prices(model, past_gen.last_sequence, 5, past_gen.scaler)
    predicted_1 = predict_future_prices(model, past_gen.last_sequence, 1, past_gen.scaler)
    future_actual = yf.download(ticker, start=test_middle, end=test_end)['Close'].values

    min_len = min(len(predicted_10), len(future_actual))
    predicted_10, predicted_5, predicted_1 = predicted_10[:min_len], predicted_5[:min_len], predicted_1[:min_len]
    future_actual = future_actual[:min_len]

    for pred, label in zip([predicted_10, predicted_5, predicted_1], ['10', '5', '1']):
        rmse = np.sqrt(mean_squared_error(future_actual, pred))
        print(f"RMSE for sentiment {label}: {rmse:.2f}")

    plt.figure(figsize=(12, 5))
    plt.plot(future_actual, label='Actual Price')
    plt.plot(predicted_10, '--', label='Predicted - 10')
    plt.plot(predicted_5, '--', label='Predicted - 5')
    plt.plot(predicted_1, '--', label='Predicted - 1')
    plt.legend()
    plt.title('NVDA 8-Month Backtest')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    news_sentiment = 5
    main(news_sentiment)
