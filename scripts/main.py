from train.train_model import train_model

def main():
    stock_ticker = input("Enter the stock ticker: ")
    try:
        news_sentiment = float(input("Rate the news sentiment (1–10): "))
        if not 1 <= news_sentiment <= 10:
            raise ValueError("Sentiment must be between 1 and 10.")
    except ValueError as e:
        print(f"Invalid input: {e}")
        return
    train_model(stock_ticker, news_sentiment)

if __name__ == "__main__":
    main()
