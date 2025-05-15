import numpy as np
import math

TIME_STEP = 60
FUTURE_DAYS = 60

def collect_data(generator):
    X_all, Y_all = [], []
    for X_batch, Y_batch in generator:
        X_all.append(X_batch)
        Y_all.append(Y_batch)
    return np.vstack(X_all), np.hstack(Y_all)

def predict_future_prices(model, last_sequence, sentiment_value, scaler, steps=FUTURE_DAYS):
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(steps):
        sentiment_effect = 0
        input_seq = current_seq.reshape(1, TIME_STEP, current_seq.shape[1])
        if sentiment_value != 5:
            if sentiment_value > 5:
                sentiment_effect = math.log(sentiment_value) / 2
            elif sentiment_value < 5:
                sentiment_effect = -1 * (1 - (math.log(sentiment_value + 1) / 2))
            pred = model.predict(input_seq, verbose=0)[0][0] * (1 + sentiment_effect)
        else:
            pred = model.predict(input_seq, verbose=0)[0][0]

        predictions.append(pred)
        next_row = np.zeros((1, current_seq.shape[1]))
        next_row[0, 0:4] = current_seq[-1, 0:4] * (1 + np.random.normal(0, 0.001, size=4))
        current_seq = np.vstack((current_seq[1:], next_row))

    dummy = np.zeros((steps, 4))
    predicted_prices = scaler.inverse_transform(np.hstack((np.array(predictions).reshape(-1, 1), dummy)))[:, 0]
    return predicted_prices
