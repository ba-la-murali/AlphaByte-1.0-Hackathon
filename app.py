import streamlit as st
from backend.fetch_stock_info import Analyze_stock
from fpdf import FPDF
import requests
import os
from dotenv import load_dotenv, dotenv_values
# Custom CSS for setting gradient background and styling
st.markdown(
    """
    <style>
    body {
        background: #34495e;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: white;
    }
    
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-color: #3498db;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #D3D3D3;
    }
    .stRadio>div>div>div>label>div:first-child {
        color: white;
        font-weight: bold;
    }
    .stRadio>div>div>div>label>div:last-child {
        color: #bdc3c7;
    }
    .stRadio>div>div>div>label:hover {
        background-color: #2c3e50;
    }
    .stRadio>div>div>div>label:active {
        background-color: #2980b9;
    }
    </style>
    """,
    unsafe_allow_html=True
)
load_dotenv()
stock_keys = os.getenv("stock_keys")
def get_financial_news(query):
    api_key = os.getenv("NEWS_API")
    # Modify the query to include keywords related to finance or company stocks
    modified_query = f"{query} AND (finance OR stock OR company)"
    url = f"https://newsapi.org/v2/everything?q={modified_query}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    articles = data.get('articles', [])
    # Filter articles based on their content to ensure relevance to financial news
    financial_articles = []
    for article in articles:
        if article.get('description'):
            if 'finance' in article['description'].lower() or 'stock' in article['description'].lower():
                financial_articles.append(article)
    return financial_articles

def get_realtime_stock_data(symbol, stock_keys):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_keys}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "Global Quote" in data:
            return data["Global Quote"]
        else:
            return None
    else:
        st.error("Failed to fetch real-time data. Please try again later.")
        return None
        
        
        

# Page title and description
st.title("📈 Stock Analysis Bot")

# Input field for user's name
name = st.text_input("May I know your name ?")

# Check if name is entered before proceeding
if name:
    # Set user's name in session state
    st.session_state["user_name"] = name
    
    # Render a welcome message
    st.markdown(f"### Hello, {name}!")
    st.markdown("Please feel free to submit any questions or inquiries related to investments.")
    
    # Display chatbot
    query = st.text_input('💬 Input your investment-related query:') 
    risk_parameter = st.radio("📊 Select Risk Parameter", ["Low", "Medium", "High"], index=1)

    # Columns for buttons and output
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    # Enter button
    with col2:
        enter_button = st.button("🚀 Enter")

    with col1:
        news_button = st.button("📰 News")

    # Clear button
    with col3:
        clear_button = st.button("🔄 Clear")
    with col4:
        stock_data_button = st.button("📈Stock")

    # Analyze the query and display results if Enter button is clicked
    if enter_button:
        if query:
            with st.spinner('⏳ Gathering all required information and analyzing. Please wait...'):
                out = Analyze_stock(query, risk_parameter, name)
            st.success('✅ Done!')
            st.write(out)
            
# Generate PDF report
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            # Replace Indian Rupee symbol with a placeholder
            cleaned_out = out.replace('\u20b9', 'Rs.')  
            pdf.multi_cell(0, 10, txt=cleaned_out)
            pdf_output = pdf.output(dest='S').encode('latin-1')  # Encoding with 'latin-1'
            with col5:
                st.download_button('Download Report', pdf_output, file_name="Report.pdf", mime="application/octet-stream")

    elif news_button:
        if query:
            financial_articles = get_financial_news(query)
            if financial_articles:
                st.subheader("Financial News:")
                for article in financial_articles:
                    st.write(f"- [{article['title']}]({article['url']})")
            else:
                st.warning("No financial news articles found for the given query.")
        else:
            st.warning('⚠ Please input your query before clicking News.')
    elif stock_data_button:
            if query:
                st.write(f"Fetching real-time data for {query}...")
                realtime_data = get_realtime_stock_data(query, stock_keys)
                if realtime_data:
                    st.subheader("Real-Time Stock Data")
                    for key, value in realtime_data.items():
                        st.write(f"- {key}: {value}")
            else:
                st.warning(f"No real-time data found for symbol {query}. Please check the symbol and try again.")
    # Clear input if Clear button is clicked
    if clear_button:
        query = ''
else:
    # Display instructions to enter name
    st.write("👋 Please enter your name to continue.")
