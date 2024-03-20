import streamlit as st
import yfinance as yf
import pandas as pd

def get_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error("Error fetching stock data: {}".format(e))
        return None

def main():
    st.title('Indian Stock Market Data')

    symbol = st.text_input('Enter Stock Symbol (e.g., TCS.NS for Tata Consultancy Services):')
    start_date = st.date_input('Start Date:')
    end_date = st.date_input('End Date:')

    if st.button('Get Data'):
        if symbol:
            st.write("Fetching data for {}...".format(symbol))
            stock_data = get_stock_data(symbol, start_date, end_date)
            if stock_data is not None:
                st.write("### Stock Data")
                st.write(stock_data)
        else:
            st.warning("Please enter a stock symbol.")

if __name__ == "__main__":
    main()
