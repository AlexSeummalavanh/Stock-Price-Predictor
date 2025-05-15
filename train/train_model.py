import datetime
from data.generator import StockDataGenerator
from models.lstm_model import build_model
from utils.helpers import predict_future_prices
from utils.plotter import plot_predictions

TIME_STEP = 60
BATCH_SIZE = 30
EPOCHS = 400
PAST_DAYS = 60
MIN_REQUIRED_DAYS = TIME_STEP + 30

def train_model(stock_ticker, news_sentiment):
    now = datetime.datetime.now()
    past_start = now - datetime.timedelta(days=PAST_DAYS + MIN_REQUIRED_DAYS)
    past_end = now
    past_gen = StockDataGenerator(past_start, past_end, stock_ticker, news_sentiment, BATCH_SIZE)

    model = build_model(input_shape=(TIME_STEP, past_gen.X.shape[2]))
    model.fit(past_gen, epochs=EPOCHS)

    future_prices = predict_future_prices(model, past_gen.last_sequence, news_sentiment, past_gen.scaler)
    predicted_vals = past_gen.scaler.inverse_transform(
        np.hstack((past_gen.Y.reshape(-1, 1), np.zeros((len(past_gen.Y), 4)))
    ))[:, 0]
    actual_prices = past_gen.original_prices[TIME_STEP:]
    plot_predictions(actual_prices, predicted_vals, future_prices, start_date=past_end)