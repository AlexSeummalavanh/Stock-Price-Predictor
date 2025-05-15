import numpy as np
import math
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

TIME_STEP = 60

class StockDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, start_date, end_date, ticker, news_sentiment, batch_size):
        stock = yf.download(ticker, start=start_date, end=end_date)
        if stock.empty:
            raise ValueError(f"No data found for ticker '{ticker}'.")

        stock_data = stock[['Close', 'Volume', 'High', 'Low', 'Open']].dropna()

        if len(stock_data) < TIME_STEP + 1:
            raise ValueError(f"Not enough data to generate sequences. Required: {TIME_STEP + 1}, Got: {len(stock_data)}")

        scaler = MinMaxScaler()
        stock_data_scaled = scaler.fit_transform(stock_data)

        avg_price = np.mean(stock_data_scaled[:, 0])
        sentiment_formula = 1 + math.log(news_sentiment) * avg_price / 100

        if news_sentiment > 5:
            sentiment_effect = sentiment_formula
        elif news_sentiment < 5:
            sentiment_effect = -sentiment_formula
        else:
            sentiment_effect = 0

        sentiment_column = np.full((stock_data_scaled.shape[0], 1), sentiment_effect)
        features = np.hstack((stock_data_scaled[:, 1:], sentiment_column))
        labels = stock_data_scaled[:, 0]

        self.scaler = scaler
        self.original_prices = stock['Close'].values
        self.last_sequence = features[-TIME_STEP:]

        self.X, self.Y = [], []
        for i in range(TIME_STEP, len(features)):
            self.X.append(features[i - TIME_STEP:i])
            self.Y.append(labels[i])
        self.X, self.Y = np.array(self.X), np.array(self.Y)

        self.batch_size = batch_size
        self.indexes = np.arange(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.X[batch_indexes], self.Y[batch_indexes]