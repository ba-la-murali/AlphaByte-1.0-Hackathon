import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download Vader lexicon if not already downloaded
import nltk
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer only once
sid = SentimentIntensityAnalyzer()

@st.cache_resource
def get_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error("Error fetching stock data: {}".format(e))
        return None

def get_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

def main():
    st.title('Indian Stock Market Data')

    symbol = st.text_input('Enter Stock Symbol (e.g., TCS.NS for Tata Consultancy Services):')
    start_date = st.date_input('Start Date:')
    end_date = st.date_input('End Date:')

    if st.button('Get Data'):
        if symbol:
            st.write("Fetching data for {}...".format(symbol))
            with st.spinner("Fetching data..."):
                stock_data = get_stock_data(symbol, start_date, end_date)
            if stock_data is not None:
                st.write("### Stock Data")
                st.write(stock_data)

                # Plotting stock price
                st.subheader("Stock Price Over Time")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(stock_data['Close'])
                ax.set_xlabel('Date')
                ax.set_ylabel('Price (INR)')
                ax.set_title('Stock Price Over Time')
                st.pyplot(fig)

                # Sentiment analysis
                st.subheader("Sentiment Analysis")
                sentiment_text = st.text_area("Enter text for sentiment analysis (e.g., news article, tweet):")
                if st.button("Analyze Sentiment"):
                    if sentiment_text:
                        sentiment_scores = get_sentiment(sentiment_text)
                        st.write("### Sentiment Scores")
                        st.write(sentiment_scores)
                        # You can display the sentiment scores in a more user-friendly format if desired

        else:
            st.warning("Please enter a stock symbol.")

if __name__ == "__main__":
    main()
