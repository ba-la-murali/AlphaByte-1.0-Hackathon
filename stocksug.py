import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import time

def get_realtime_prices(stocks):
    prices = {}
    for stock_symbol in stocks:
        stock = yf.Ticker(stock_symbol)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        prices[stock_symbol] = current_price
    return prices

def get_recommendation(investment_amount, stocks, risk_factor):
    recommendations = {}
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    for stock_symbol in stocks:
        stock_data = yf.download(stock_symbol, start="2020-01-01", end=end_date)
        stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
        avg_daily_return = stock_data['Daily_Return'].mean()
        std_dev_daily_return = stock_data['Daily_Return'].std()
        
        if risk_factor == "Low":
            risk_threshold = 0.05
        elif risk_factor == "Medium":
            risk_threshold = 0.15
        elif risk_factor == "High":
            risk_threshold = 0.25
        
        if avg_daily_return > 0 and std_dev_daily_return < risk_threshold:
            recommendations[stock_symbol] = {"Recommendation": "Buy", "Current_Price": stock_data['Adj Close'].iloc[-1]}
        else:
            recommendations[stock_symbol] = {"Recommendation": "Hold", "Current_Price": stock_data['Adj Close'].iloc[-1]}
    
    return recommendations

st.title("Stock Recommendation App")

investment_amount = st.number_input("Enter the amount you want to invest:", min_value=1, step=1)
stock_symbols = st.text_input("Enter comma-separated list of stock symbols (e.g., AAPL, MSFT, GOOGL):")
risk_factor = st.selectbox("Choose the risk factor:", ["Low", "Medium", "High"])

if st.button("Get Recommendations"):
    stocks = [symbol.strip() for symbol in stock_symbols.split(",")]
    recommendations = get_recommendation(investment_amount, stocks, risk_factor)
    
    st.write("Recommendations:")
    for stock_symbol, data in recommendations.items():
        st.write(f"{stock_symbol}: {data['Recommendation']}")

    st.write("\nReal-time Prices:")
    realtime_prices = get_realtime_prices(stocks)
    for stock_symbol, price in realtime_prices.items():
        st.write(f"{stock_symbol}: {price}")
