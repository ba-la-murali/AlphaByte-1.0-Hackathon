# AB-05 - Problem Statement
## StockSavvy - Stock Analysis Bot 

## Description: 
This repository contains code for a Stock Analysis Bot built using Streamlit and Python. The bot provides users with investment-related insights, fetches financial news, and generates reports based on user queries.

## Features:
- Fetch financial news using the News API.
- Analyze user queries related to investments using OpenAI's GPT-3.5.
- Display stock data from Yahoo Finance.
- Generate PDF reports with analysis results.

## Prerequisites:
- Python 3.x
- Install required dependencies using `pip install -r requirements.txt`.
- Set up environment variables for API keys.

## How to Use:
1. Clone this repository to your local machine.
2. Install dependencies using `pip install -r requirements.txt`.
3. Rename .env.example to .env and fill in your API keys appropriately in the .env file.
4. Set up environment variables for API keys:
   - `NEWS_API`: News API key.
   - `OPEN_AI_KEY`: OpenAI API key.
5. Run the application using `streamlit run app.py`.
6. Enter your name, investment queries, and risk parameters.
7. Click on buttons to fetch news, clear inputs, or generate reports.
8. Download PDF reports with analysis results.

## Directory Structure:
- `app.py`: Main script for the Streamlit web application.
- `backend/fetch_stock_info.py`: Backend script for fetching stock data and performing analysis.
- `.env`: Environment variables file for API keys.
- `requirements.txt`: List of dependencies.

## Credits:
- Streamlit: [streamlit.io](https://streamlit.io/)
- News API: [newsapi.org](https://newsapi.org/)
- OpenAI: [openai.com](https://openai.com/)
- Yahoo Finance: [finance.yahoo.com](https://finance.yahoo.com/)

# Thank you for using StockSavvy!

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
