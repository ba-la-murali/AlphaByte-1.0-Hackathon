import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
import plotly.io as pio

from nltk.sentiment.vader import SentimentIntensityAnalyzer

 
# import nltk
# nltk.download('vader_lexicon')
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# add_bg_from_local('img.gif')
# Initialize the sentiment analyzer only once
sid = SentimentIntensityAnalyzer()

@st.cache_data
def get_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error("Error fetching stock data for {}: {}".format(symbol, e))
        return None

def plot_candlestick_chart(stock_data):
    fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'])])

    fig.update_layout(title='Candlestick Chart',
                      xaxis_title='Date',
                      yaxis_title='Price (INR)',
                      xaxis_rangeslider_visible=False)

    return fig

def plot_line_chart(stock_data, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name=symbol))

    fig.update_layout(title='Stock Price Comparison',
                      xaxis_title='Date',
                      yaxis_title='Price (INR)')

    return fig

def main():
    st.title('📈TimeLine Data Comparison📈')

    symbols = st.text_input('Enter Stock Symbols (e.g., TCS.NS, INFY.NS):').split(',')
    start_date = st.date_input('Start Date:')
    end_date = st.date_input('End Date:')

    if st.button('Get Data'):
        with st.spinner('⏳ Gathering all required information and analyzing. Please wait...'):
            if symbols:
                fig_comparison = go.Figure() 
                for symbol in symbols:
                    st.write("Fetching data for {}...".format(symbol))
                    with st.spinner("Fetching data..."):
                        stock_data = get_stock_data(symbol.strip(), start_date, end_date)
                    if stock_data is not None:
                        st.write("### Stock Data for {}".format(symbol))
                        st.write(stock_data)

                        # Plotting stock price
                        st.subheader("Stock Price Over Time")
                        fig_stock_price, ax_stock_price = plt.subplots(figsize=(10, 6))
                        ax_stock_price.plot(stock_data['Close'])
                        ax_stock_price.set_xlabel('Date')
                        ax_stock_price.set_ylabel('Price (INR)')
                        ax_stock_price.set_title('Stock Price Over Time')
                        st.pyplot(fig_stock_price)

                        # Plotting candlestick chart
                        st.subheader("Candlestick Chart")
                        st.write("This chart shows the open, high, low, and close prices of the stock over time.")
                        fig_candlestick = plot_candlestick_chart(stock_data)
                        st.plotly_chart(fig_candlestick)

                        # Download button for candlestick chart
                        st.markdown(get_chart_download_link(fig_candlestick, f"{symbol}_Candlestick_Chart_Report.html"), unsafe_allow_html=True)

    # Plotting line chart
                        fig_comparison.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name=symbol))

                        st.subheader("Stock Price Comparison - Line Chart")
                        st.write("This chart compares the closing prices of the selected companies over time.")
                        st.plotly_chart(fig_comparison)
                        # Download button for comparison chart
                        st.markdown(get_chart_download_link(fig_comparison, "Companies_Comparison_Line_Chart_Report.html"), unsafe_allow_html=True) 

            else:
                st.warning("Please enter at least one stock symbol.")

def get_chart_download_link(figure, filename, linktext="Download HTML"):
    fig_data = pio.to_html(figure, full_html=False)
    b64 = base64.b64encode(fig_data.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">{linktext}</a>'
    return href

if __name__ == "__main__":
    main()