import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import time
import os
from huggingface_hub import InferenceClient

class HuggingFaceLlamaLLM:
    def __init__(self, api_key, model_name="meta-llama/Llama-3.1-8B-Instruct", max_tokens=1000, temperature=0):
        self.client = InferenceClient(
            provider="fireworks-ai",
            api_key=api_key,
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def __call__(self, prompt):
        return self.generate(prompt)
    
    def generate(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error calling Hugging Face API: {str(e)}")
            return "Sorry, I encountered an error processing your request."

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

# Initialize Llama model
HF_TOKEN = os.getenv("HF_TOKEN")
llm = HuggingFaceLlamaLLM(api_key=HF_TOKEN, temperature=0, max_tokens=1000)

investment_amount = st.number_input("Enter the amount you want to invest:", min_value=1, step=1)
stock_symbols = st.text_input("Enter comma-separated list of stock symbols (e.g., RELIANCE.NS,TCS.NS):")
risk_factor = st.selectbox("Choose the risk factor:", ["Low", "Medium", "High"])

if st.button("Get Recommendations"):
    stocks = [symbol.strip() for symbol in stock_symbols.split(",")]
    recommendations = get_recommendation(investment_amount, stocks, risk_factor)
    st.write("\nREAL TIME DATA\n")
    for stock_symbol, data in recommendations.items():
        st.write(f"{stock_symbol}: {data['Recommendation']}")

    st.write("\nReal-time Prices:")
    realtime_prices = get_realtime_prices(stocks)
    for stock_symbol, price in realtime_prices.items():
        st.write(f"{stock_symbol}: {price}")
    
    prompt = f"""Give detail stock analysis, Use the available data and provide investment recommendation. You have the following information available about the stocks {recommendations}. Don't show price of any stock. User has selected {risk_factor}. Write (5-6) lines investment analysis to answer user query, At the start itself give recommendation to user about the stock."""
    
    analysis = llm(prompt)
    st.write("\nCONCLUSION\n") 
    st.write(analysis)
