import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
import plotly.io as pio
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer only once
sid = SentimentIntensityAnalyzer()

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

@st.cache_data
def get_stock_data(symbol, start_date, end_date):
    try:
        # Ensure we get a clean DataFrame with proper column names
        stock_data = yf.download(symbol, start=start_date, end=end_date, group_by='ticker')
        
        # If multi-level columns, flatten them
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[1] if col[1] else col[0] for col in stock_data.columns]
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        
        if missing_columns:
            st.error(f"Missing columns for {symbol}: {missing_columns}")
            return None
            
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {e}")
        return None

def plot_candlestick_chart(stock_data, symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name=symbol
    )])

    fig.update_layout(
        title=f'Candlestick Chart - {symbol}',
        xaxis_title='Date',
        yaxis_title='Price (INR)',
        xaxis_rangeslider_visible=False,
        height=600
    )

    return fig

def plot_line_chart(stock_data_dict):
    fig = go.Figure()
    
    for symbol, data in stock_data_dict.items():
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['Close'], 
            mode='lines', 
            name=symbol,
            line=dict(width=2)
        ))

    fig.update_layout(
        title='Stock Price Comparison - Line Chart',
        xaxis_title='Date',
        yaxis_title='Price (INR)',
        height=600,
        hovermode='x unified'
    )

    return fig

def get_chart_download_link(figure, filename, linktext="Download HTML"):
    fig_data = pio.to_html(figure, full_html=False)
    b64 = base64.b64encode(fig_data.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">{linktext}</a>'
    return href

def main():
    st.title('📈TimeLine Data Comparison📈')

    symbols_input = st.text_input('Enter Stock Symbols (e.g., TCS.NS, INFY.NS):')
    start_date = st.date_input('Start Date:')
    end_date = st.date_input('End Date:')

    if st.button('Get Data'):
        if symbols_input.strip():
            symbols = [symbol.strip() for symbol in symbols_input.split(',') if symbol.strip()]
            
            with st.spinner('⏳ Gathering all required information and analyzing. Please wait...'):
                stock_data_dict = {}
                
                # Fetch data for all symbols
                for symbol in symbols:
                    st.write(f"Fetching data for {symbol}...")
                    stock_data = get_stock_data(symbol, start_date, end_date)
                    
                    if stock_data is not None and not stock_data.empty:
                        stock_data_dict[symbol] = stock_data
                        
                        # Display individual stock data
                        st.write(f"### Stock Data for {symbol}")
                        st.dataframe(stock_data.tail(10))  # Show last 10 rows
                        
                        # Individual stock price chart using matplotlib
                        st.subheader(f"Stock Price Over Time - {symbol}")
                        fig_stock_price, ax_stock_price = plt.subplots(figsize=(12, 6))
                        ax_stock_price.plot(stock_data.index, stock_data['Close'], linewidth=2)
                        ax_stock_price.set_xlabel('Date')
                        ax_stock_price.set_ylabel('Price (INR)')
                        ax_stock_price.set_title(f'Stock Price Over Time - {symbol}')
                        ax_stock_price.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_stock_price)
                        plt.close()
                        
                        # Individual candlestick chart
                        st.subheader(f"Candlestick Chart - {symbol}")
                        st.write("This chart shows the open, high, low, and close prices of the stock over time.")
                        fig_candlestick = plot_candlestick_chart(stock_data, symbol)
                        st.plotly_chart(fig_candlestick, use_container_width=True)
                        
                        # Download button for individual candlestick chart
                        st.markdown(
                            get_chart_download_link(
                                fig_candlestick, 
                                f"{symbol}_Candlestick_Chart_Report.html"
                            ), 
                            unsafe_allow_html=True
                        )
                        
                        st.markdown("---")  # Add separator
                
                # Comparison chart (only if we have multiple stocks)
                if len(stock_data_dict) > 1:
                    st.subheader("📊 Stock Price Comparison - All Symbols")
                    st.write("This chart compares the closing prices of all selected companies over time.")
                    
                    fig_comparison = plot_line_chart(stock_data_dict)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Download button for comparison chart
                    st.markdown(
                        get_chart_download_link(
                            fig_comparison, 
                            "Companies_Comparison_Line_Chart_Report.html"
                        ), 
                        unsafe_allow_html=True
                    )
                elif len(stock_data_dict) == 1:
                    st.info("Add more symbols to see a comparison chart.")
                else:
                    st.error("No valid stock data was retrieved. Please check your symbols and date range.")
        else:
            st.warning("Please enter at least one stock symbol.")

if __name__ == "__main__":
    main()
