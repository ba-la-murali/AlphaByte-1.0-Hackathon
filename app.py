import streamlit as st
from backend.fetch_stock_info import Analyze_stock
from fpdf import FPDF
import requests
import os
from dotenv import load_dotenv, dotenv_values

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
    modified_query = f"{query} AND (finance OR stock OR company)"
    url = f"https://newsapi.org/v2/everything?q={modified_query}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    articles = data.get('articles', [])
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

st.title("📈 StockSavvy - Stock Analysis Bot")
name = st.text_input("May I know your name ?")

if name:
    st.session_state["user_name"] = name
    st.markdown(f"### Hello, {name}!")
    st.markdown("Please feel free to submit any questions or inquiries related to investments.")
    query = st.text_input('💬 Input your investment-related query:') 
    risk_parameter = st.radio("📊 Select Risk Parameter", ["Low", "Medium", "High"], index=1)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col2:
        enter_button = st.button("🚀 Enter")

    with col1:
        news_button = st.button("📰 News")

    with col3:
        clear_button = st.button("🔄 Clear")
    with col4:
        stock_data_button = st.button("📈Stock")

    if enter_button:
        if query:
            with st.spinner('⏳ Gathering all required information and analyzing. Please wait...'):
                out = Analyze_stock(query, risk_parameter, name)
            st.success('✅ Done!')
            st.write(out)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            cleaned_out = out.replace('\u20b9', 'Rs.')  
            pdf.multi_cell(0, 10, txt=cleaned_out)
            pdf_output = pdf.output(dest='S').encode('latin-1') 
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

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, txt=f"Real-Time Stock Data for {query}:\n")
                for key, value in realtime_data.items():
                    pdf.cell(0, 10, txt=f"- {key}: {value}", ln=True)
                pdf_output = pdf.output(dest='S').encode('latin-1') 
                with col5:
                    st.download_button('Download Stock Data Report', pdf_output, file_name="Stock_Data_Report.pdf", mime="application/octet-stream")
            else:
                st.warning(f"No real-time data found for symbol {query}. Please check the symbol and try again.")
        else:
            st.warning("Please enter a symbol before fetching real-time stock data.")

    if clear_button:
        query = ''
else:
    st.write("👋 Please enter your name to continue.")
