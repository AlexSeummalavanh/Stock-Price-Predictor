import matplotlib.pyplot as plt
import pandas as pd
import datetime

def plot_predictions(past_vals, predicted_vals, future_vals=None, start_date=None):
    plt.figure(figsize=(14, 6))
    total_days = len(past_vals) + len(predicted_vals) + (len(future_vals) if future_vals is not None else 0)
    if start_date:
        dates = pd.date_range(start=start_date - datetime.timedelta(days=len(past_vals)), periods=total_days, freq='B')
    else:
        dates = list(range(total_days))

    plt.plot(dates[:len(past_vals)], past_vals, label='Past Price')

    if future_vals is not None:
        future_start = len(past_vals)
        plt.plot(dates[future_start:future_start+len(future_vals)], future_vals, label='Future Forecast')

    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.title("Stock Price Prediction")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()