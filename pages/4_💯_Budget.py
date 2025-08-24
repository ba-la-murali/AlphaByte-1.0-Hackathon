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
        try:
            stock = yf.Ticker(stock_symbol)
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prices[stock_symbol] = current_price
            else:
                prices[stock_symbol] = "No data"
        except Exception as e:
            prices[stock_symbol] = f"Error: {str(e)}"
    return prices

def get_recommendation(investment_amount, stocks, risk_factor):
    recommendations = {}
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    for stock_symbol in stocks:
        try:
            # Download stock data
            stock_data = yf.download(stock_symbol, start="2020-01-01", end=end_date, progress=False)
            
            # Check if data was downloaded successfully
            if stock_data.empty:
                st.warning(f"No data found for {stock_symbol}. Please check the symbol.")
                recommendations[stock_symbol] = {"Recommendation": "No Data", "Current_Price": "N/A"}
                continue
            
            # Handle MultiIndex columns
            if isinstance(stock_data.columns, pd.MultiIndex):
                # Flatten the MultiIndex by taking the first level (the actual column names)
                stock_data.columns = stock_data.columns.droplevel(1)
            
            # Check available columns and use Close instead of Adj Close
            if 'Close' in stock_data.columns:
                stock_data['Daily_Return'] = stock_data['Close'].pct_change()
                current_price = stock_data['Close'].iloc[-1]
            elif 'Adj Close' in stock_data.columns:
                stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
                current_price = stock_data['Adj Close'].iloc[-1]
            else:
                st.error(f"No suitable price column found for {stock_symbol}")
                recommendations[stock_symbol] = {"Recommendation": "Data Error", "Current_Price": "N/A"}
                continue
            
            # Calculate metrics
            avg_daily_return = stock_data['Daily_Return'].mean()
            std_dev_daily_return = stock_data['Daily_Return'].std()
            
            # Set risk thresholds
            if risk_factor == "Low":
                risk_threshold = 0.05
            elif risk_factor == "Medium":
                risk_threshold = 0.15
            elif risk_factor == "High":
                risk_threshold = 0.25
            
            # Make recommendation
            if avg_daily_return > 0 and std_dev_daily_return < risk_threshold:
                recommendations[stock_symbol] = {"Recommendation": "Buy", "Current_Price": current_price}
            else:
                recommendations[stock_symbol] = {"Recommendation": "Hold", "Current_Price": current_price}
                
        except Exception as e:
            st.error(f"Error processing {stock_symbol}: {str(e)}")
            recommendations[stock_symbol] = {"Recommendation": "Error", "Current_Price": "N/A"}
    
    return recommendations

st.title("Stock Recommendation App")

# Initialize Llama model
HF_TOKEN = os.getenv("HF_TOKEN")
llm = HuggingFaceLlamaLLM(api_key=HF_TOKEN, temperature=0, max_tokens=1000)

investment_amount = st.number_input("Enter the amount you want to invest:", min_value=1, step=1)
stock_symbols = st.text_input("Enter comma-separated list of stock symbols (e.g., RELIANCE.NS,TCS.NS):")
risk_factor = st.selectbox("Choose the risk factor:", ["Low", "Medium", "High"])

if st.button("Get Recommendations"):
    if not stock_symbols.strip():
        st.error("Please enter at least one stock symbol.")
    else:
        stocks = [symbol.strip().upper() for symbol in stock_symbols.split(",")]
        
        with st.spinner("Fetching stock data and generating recommendations..."):
            recommendations = get_recommendation(investment_amount, stocks, risk_factor)
            
            st.write("\n**REAL TIME DATA**\n")
            for stock_symbol, data in recommendations.items():
                st.write(f"**{stock_symbol}**: {data['Recommendation']}")

            st.write("\n**Real-time Prices:**")
            realtime_prices = get_realtime_prices(stocks)
            for stock_symbol, price in realtime_prices.items():
                if isinstance(price, (int, float)):
                    st.write(f"**{stock_symbol}**: ₹{price:.2f}")
                else:
                    st.write(f"**{stock_symbol}**: {price}")
            
            # Generate AI analysis
            prompt = f"""Give detailed stock analysis. Use the available data and provide investment recommendation. You have the following information available about the stocks {recommendations}. Don't show price of any stock. User has selected {risk_factor} risk factor. Write (5-6) lines investment analysis to answer user query. At the start itself give recommendation to user about the stock."""
            
            analysis = llm(prompt)
            st.write("\n**CONCLUSION**\n") 
            st.write(analysis)
