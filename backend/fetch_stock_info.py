import json
import time
from bs4 import BeautifulSoup
import re
import requests
from dotenv import load_dotenv
import yfinance as yf
import warnings
import os
from huggingface_hub import InferenceClient

warnings.filterwarnings("ignore")

# Load variables from .env file
load_dotenv()

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
            print(f"Error calling Hugging Face API: {str(e)}")
            return "Sorry, I encountered an error processing your request."

# Initialize Llama model
HF_TOKEN = os.getenv("HF_TOKEN")
llm = HuggingFaceLlamaLLM(api_key=HF_TOKEN, temperature=0, max_tokens=1000)

def get_stock_price(ticker, history=5):
    if "." in ticker:
        ticker = ticker.split(".")[0]
    ticker = ticker + ".NS"
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    df = df[["Close", "Volume"]]
    df.index = [str(x).split()[0] for x in list(df.index)]
    df.index.rename("Date", inplace=True)
    df = df[-history:]
    return df.to_string()

def google_query(search_term):
    if "news" not in search_term:
        search_term = search_term + " stock news"
    url = f"https://www.google.com/search?q={search_term}&cr=countryIN"
    url = re.sub(r"\s", "+", url)
    return url

def get_recent_stock_news(company_name):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    g_query = google_query(company_name)
    res = requests.get(g_query, headers=headers).text
    soup = BeautifulSoup(res, "html.parser")
    news = []
    for n in soup.find_all("div", "n0jPhd ynAwRc tNxQIb nDgy9d"):
        news.append(n.text)
    for n in soup.find_all("div", "IJl0Z"):
        news.append(n.text)

    if len(news) > 6:
        news = news[:4]
    news_string = ""
    for i, n in enumerate(news):
        news_string += f"{i}. {n}\n"
    top5_news = "Recent News:\n\n" + news_string
    return top5_news

def get_financial_statements(ticker):
    if "." in ticker:
        ticker = ticker.split(".")[0]
    ticker = ticker + ".NS"    
    company = yf.Ticker(ticker)
    balance_sheet = company.balance_sheet
    if balance_sheet.shape[1] >= 3:
        balance_sheet = balance_sheet.iloc[:, :3]
    balance_sheet = balance_sheet.dropna(how="any")
    balance_sheet = balance_sheet.to_string()
    return balance_sheet

def get_stock_ticker(query):
    """
    Extract company name and ticker from query using Llama
    """
    client = InferenceClient(
        provider="fireworks-ai",
        api_key=HF_TOKEN,
    )
    
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Indian stock markets. Extract company name and ticker symbol from user queries about Indian stocks (NSE/BSE)."
                },
                {
                    "role": "user",
                    "content": f"""Extract the company name and stock ticker symbol from this query about Indian stocks: "{query}"

Respond in this exact JSON format:
{{"company_name": "company name here", "ticker_symbol": "ticker symbol here"}}

For Indian stocks, add .NS suffix if not present. For example:
- Reliance -> RELIANCE.NS
- TCS -> TCS.NS
- HDFC -> HDFC.NS

Query: {query}"""
                }
            ],
            max_tokens=200,
            temperature=0
        )
        
        response = completion.choices[0].message.content
        
        # Try to parse JSON response
        result = json.loads(response)
        company_name = result.get("company_name", "")
        ticker_symbol = result.get("ticker_symbol", "")
        return company_name, ticker_symbol
        
    except Exception as e:
        print(f"Error in get_stock_ticker: {str(e)}")
        # Fallback: try to extract from query directly
        words = query.upper().split()
        for word in words:
            if word in ['TCS', 'RELIANCE', 'HDFC', 'ICICI', 'INFY', 'WIPRO']:
                return word, f"{word}.NS"
        return "Unknown Company", "UNKNOWN.NS"

def Analyze_stock(query, risk, name):
    Company_name, ticker = get_stock_ticker(query)
    print({"Query": query, "Company_name": Company_name, "Ticker": ticker})
    
    stock_data = get_stock_price(ticker, history=10)
    stock_financials = get_financial_statements(ticker)
    stock_news = get_recent_stock_news(Company_name)

    available_information = f"Stock Price: {stock_data}\n\nStock Financials: {stock_financials}\n\nStock News: {stock_news}"

    prompt = f"""Give detail stock analysis, Use the available data and provide investment recommendation. At the start itself give conclusion to user about the stock User's Name is {name}. The user is fully aware about the investment risk, dont include any kind of warning like 'It is recommended to conduct further research and analysis or consult with a financial advisor before making an investment decision' in the answer The user is interested in investments having risk tolerance : {risk}. User question: {query}. You have the following information available about {Company_name}. Write (5-8) pointwise investment analysis to answer user query, At the start itself give recommendation to user about the stock. {available_information}"""
    
    analysis = llm(prompt)
    return analysis
