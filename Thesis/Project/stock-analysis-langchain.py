import os
import yfinance as yf
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
# from langchain.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Qdrant
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.llms import OpenAI as LangchainOpenAI
import logging
from transformers import pipeline
import numpy as np
import statsmodels.api as sm
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
from cleanlab_studio import Studio

# Logging setup
logging.basicConfig(level=logging.INFO)
logging.info("Imported all modules")

RISK_FREE_RATE = 0.02  # Risk-free rate for Sharpe ratio

collection_name = "financial_documents"

logging.info("Initialized all variables")

# Qdrant setup
# qdrant_client = QdrantClient(path="financial_analysis")
OPENAI_API_KEY = "3a6b230b917b4893a150f0ad7fa126cf"
os.environ["AZURE_OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://cpe-clx-openai.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-05-15" 
studio = Studio("94ae2b40d9414d4b873b1a94d3da5999")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Azure OpenAI setup for LLM and embeddings
# llm = AzureOpenAI(
#     openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
#     openai_api_version=os.environ["OPENAI_API_VERSION"],
#     # azure_openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     model="gpt-4o-2024-05-13"
# )

llm = AzureChatOpenAI(
    model="cpe-clx-gpt4o",
    azure_deployment="cpe-clx-gpt4o",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

# embedding_model = AzureOpenAIEmbeddings(
#     openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
#     azure_openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     model="text-embedding-ada-002"
# )

embedding_model = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    # deployment_name="cpe-clx-embedding",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"] ,
    azure_deployment="cpe-clx-embedding"
)

tlm = studio.TLM() 

logging.info("Initialized Azure OpenAI models")

# Updated function to parse and vectorize financial documents using LangChain DirectoryLoader
# def process_financial_documents(folder_path):

#     company_vectors = {}

#     if qdrant_client.collection_exists(collection_name=collection_name):
#         logging.info("Using existing Qdrant collection")

#         vector_store = Qdrant(
#             client=qdrant_client,
#             collection_name=collection_name,
#             embeddings=embedding_model
#         )

#         company_vectors['ADS.DE'] = vector_store

#     else:
#         qdrant_client.recreate_collection(
#             collection_name=collection_name,
#             vectors_config=VectorParams(size=1536, distance=Distance.COSINE)  # Assuming vector size of 1536
#         )
#         logging.info("Created new Qdrant collection")

#         for company_name in os.listdir(folder_path):
#             company_folder = os.path.join(folder_path, company_name)

#             logging.info(f"Processing documents for {company_name}")
#             if os.path.isdir(company_folder):
#                 loader = DirectoryLoader(company_folder, recursive=True)
#                 documents = loader.load()

#                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                 docs = text_splitter.split_documents(documents)

#                 vector_store = Qdrant.from_documents(
#                     documents=docs,
#                     embedding=embedding_model,
#                     client=qdrant_client,
#                     collection_name=collection_name
#                 )
                
#                 company_vectors[company_name] = vector_store

#     return vector_store

# Scraping stock prices, volatility, and PE ratio from Yahoo Finance
def get_stock_data(company_name, num_months):

    # information of stock
    stock = yf.Ticker(company_name)

    # benchmark of listed stockmarket
    # nifty 50 index for beta calculation - ^NSEI - India
    # S&P 500 Index for beta calculation - ^GSPC - USA
    # DAX Index for beta calculation - ^GDAXI - Germany
    benchmark = yf.Ticker("^NSEI")  # S&P 500 Index for beta calculation

    # timeline to analyze
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * num_months)

    # history of the stock
    stock_hist = stock.history(start=start_date, end=end_date)
    benchmark_hist = benchmark.history(start=start_date, end=end_date)

    # stock price history json
    stock_history = stock_hist.to_json()

    # Calculate daywise returns trend of the stock
    stock_returns = stock_hist['Close'].pct_change().dropna()

    # Calculate daywise returns trend of the benchmark
    benchmark_returns = benchmark_hist['Close'].pct_change().dropna()

    # Algining the stock and benchmark returns
    stock_returns, benchmark_returns = stock_returns.align(benchmark_returns, join='inner')

    # Volatility calculation
    volatility = stock_returns.std() * np.sqrt(252)  # Annualized volatility

    # PE ratio
    pe_ratio = stock.info.get('trailingPE', None)

    # Financial metrics calculation (Alpha, Beta, R-squared, Std Deviation, Sharpe Ratio)
    X = sm.add_constant(benchmark_returns)
    model = sm.OLS(stock_returns, X).fit()
    beta = model.params[1]  # Beta
    alpha = model.params[0]  # Alpha
    r_squared = model.rsquared  # R-squared
    std_dev = stock_returns.std()  # Standard deviation of stock returns

    # Sharpe ratio
    mean_return = stock_returns.mean() * 252  # Annualized return
    sharpe_ratio = (mean_return - RISK_FREE_RATE) / std_dev

    # Scrape news for sentiment analysis
    news_url = f"https://finance.yahoo.com/lookup/?s={company_name}"
    news_content = requests.get(news_url)
    soup = BeautifulSoup(news_content.content, 'html.parser')
    news_articles = [h3.get_text() for h3 in soup.find_all('h3')]
    news_texts = [heading for heading in news_articles if len(heading.split()) > 4]

    # stock income statements
    stock_income_stmt = stock.income_stmt
    stock_income_stmt.columns = stock_income_stmt.columns.astype(str)
    income_statement = stock_income_stmt.to_json() 

    # stock balance sheet
    stock_balance_sheet = stock.balance_sheet
    stock_balance_sheet.columns = stock_balance_sheet.columns.astype(str)
    balance_sheet = stock_balance_sheet.to_json()

    # stock cash flow
    stock_cash_flow = stock.cashflow
    stock_cash_flow.columns = stock_cash_flow.columns.astype(str)
    cash_flow = stock_cash_flow.to_json()

    eps_trend_90day = stock.eps_trend['90daysAgo']
    eps_trend_60day = stock.eps_trend['60daysAgo']
    eps_trend_30day = stock.eps_trend['30daysAgo']
    eps_trend_current = stock.eps_trend['current']

    growth_estimates = stock.growth_estimates['stock']


    return {
        "eps_trend_90day": eps_trend_90day,
        "eps_trend_60day": eps_trend_60day,
        "eps_trend_30day": eps_trend_30day,
        "eps_trend_current": eps_trend_current,
        "growth_estimate": growth_estimates,
        "volatility": volatility,
        "pe_ratio": pe_ratio,
        "beta": beta,
        "alpha": alpha,
        "r_squared": r_squared,
        "std_dev": std_dev,
        "sharpe_ratio": sharpe_ratio,
        "news_texts": news_texts
    }

# Perform sentiment analysis using HuggingFace transformers
def get_sentiment(news_texts):
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiments = sentiment_pipeline(news_texts)
    positive_sentiment = sum([1 for s in sentiments if s['label'] == 'POSITIVE'])
    negative_sentiment = sum([1 for s in sentiments if s['label'] == 'NEGATIVE'])

    print("Positive Sentiments: ", positive_sentiment)
    print("Negative Sentiments: ", negative_sentiment)

    total = len(sentiments)

    return (positive_sentiment - negative_sentiment) / total  # Net sentiment score

# Function to query GPT-4 with financial data and question using Azure OpenAI and LangChain
def query_gpt_4_and_tlm(company_name, stock_data, sentiment_score, risk, question):
    # Prepare context for Azure OpenAI

    context = f"""
    Company: {company_name}

    EPS Trend (90 Days Ago): {stock_data['eps_trend_90day']}
    EPS Trend (60 Days Ago): {stock_data['eps_trend_60day']}
    EPS Trend (30 Days Ago): {stock_data['eps_trend_30day']}
    EPS Trend (Current): {stock_data['eps_trend_current']}
    Growth Estimate: {stock_data['growth_estimate']}
    Volatility: {stock_data['volatility']:.2f}
    PE Ratio: {stock_data['pe_ratio']}
    Beta: {stock_data['beta']:.2f}
    Alpha: {stock_data['alpha']:.4f}
    R-squared: {stock_data['r_squared']:.2f}
    Standard Deviation: {stock_data['std_dev']:.4f}
    Sharpe Ratio: {stock_data['sharpe_ratio']:.2f}
    Sentiment Score: {sentiment_score:.2f}
    Risk: {risk}
    Question: {question}
    """
    #  Financial Document Context: {context}

    prompt_template = PromptTemplate(
        input_variables=['question'], 
        template=context + "\nQuestion: {question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    response = chain.run(question)

    # Placeholder for trustworthiness score (this would be calculated separately)
    # trustworthiness_score = np.random.uniform(0.7, 0.9)  # Placeholder for actual TLM logic
    trustworthiness_score = tlm.get_trustworthiness_score(context, response=response)


    # Print GPT-4 response and trustworthiness score
    print(f"--- GPT-4 Response for {company_name} ---")
    print(response)
    print(f"Trustworthiness Score: {trustworthiness_score['trustworthiness_score']}")

    return response, trustworthiness_score

# Function for risk analysis using LangChain's retriever and OpenAI
def risk_analysis(company_name, question):
    # Retrieve relevant company documents

    # Get stock data
    stock_data = get_stock_data(company_name, 6)  # Assume 6 months for example
    sentiment_score = get_sentiment(stock_data["news_texts"])

    # Final risk analysis
    risk = "High" if stock_data["volatility"] > 0.3 or sentiment_score < 0 else "Low"
    strategy = "Buy" if stock_data["pe_ratio"] < 15 and sentiment_score > 0 else "Hold/Sell"

    # Query GPT-4 via Azure OpenAI and get trustworthiness score
    gpt_response, trustworthiness_score = query_gpt_4_and_tlm(
        company_name, stock_data, sentiment_score, risk, question
    )

    # Print financial metrics and analysis
    # print(f"Company: {company_name}")
    # print(f"Volatility: {stock_data['volatility']:.2f}")
    # print(f"PE Ratio: {stock_data['pe_ratio']}")
    # print(f"Beta: {stock_data['beta']:.2f}")
    # print(f"Alpha: {stock_data['alpha']:.4f}")
    # print(f"R-squared: {stock_data['r_squared']:.2f}")
    # print(f"Standard Deviation: {stock_data['std_dev']:.4f}")
    # print(f"Sharpe Ratio: {stock_data['sharpe_ratio']:.2f}")
    # print(f"Sentiment Score: {sentiment_score:.2f}")
    # print(f"Risk Level: {risk}")
    # print(f"Strategy: {strategy}")
    # print(f"Trustworthiness Score: {trustworthiness_score['trustworthiness_score']}")

    return gpt_response, trustworthiness_score

# Example usage: replace 'ADS.DE' with the actual company ticker you want to analyze
folder_path = "C:\\Users\\CQTF47\\Desktop\\IU Masters\\Thesis\\Data"
# company_vectors = process_financial_documents(folder_path)
question = "Explain the Technical and Fundamental Analysis of the Stock and What are the risks associated with investing in Adidas?"

risk_analysis("ADS.DE", question)
