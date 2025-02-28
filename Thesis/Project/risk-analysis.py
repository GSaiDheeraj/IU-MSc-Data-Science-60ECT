import os
import yfinance as yf
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
# from langchain_milvus import Milvus
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
# from langchain.chains import RetrievalQAChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from transformers import pipeline
import numpy as np
import statsmodels.api as sm
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import PointStruct, VectorParams, Distance, HnswConfig,HnswConfigDiff
import logging
import fitz
import warnings

logging.basicConfig(level=logging.INFO)
logging.info("Imported all modules")

RISK_FREE_RATE = 0.02  # Risk-free rate for Sharpe ratio

logging.info("Initialized all variables")

qdrant_client = QdrantClient(path="financial_analysis")

if qdrant_client.collection_exists(collection_name="financial_documents"):
    qdrant_client.delete_collection(collection_name="financial_documents")

qdrant_client.create_collection(
    collection_name="financial_documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),  # Assuming vector size of 1536
)

logging.info("Created Qdrant collection")

OPENAI_API_KEY = "3a6b230b917b4893a150f0ad7fa126cf"
os.environ["AZURE_OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://cpe-clx-openai.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-05-15" 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

azure_llm = AzureChatOpenAI(
    model="cpe-clx-gpt4o",
    azure_deployment="cpe-clx-gpt4o",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

embed_model = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    # deployment_name="cpe-clx-embedding",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"] ,
    azure_deployment="cpe-clx-embedding"
)

logging.info("Initialized Azure OpenAI models")

vectorstore = Qdrant(client=qdrant_client, collection_name="financial_documents", embedding_function=embed_model.embed_documents)

logging.info("Initialized Qdrant vector store")

# Function to parse and vectorize financial documents
# def process_financial_documents(folder_path):
#     company_vectors = {}
#     for company_name in os.listdir(folder_path):
#         print(f"Processing documents for {company_name}")
#         company_folder = os.path.join(folder_path, company_name)
#         if os.path.isdir(company_folder):
#             documents = []
#             for file in os.listdir(company_folder):
#                 file_path = os.path.join(company_folder, file)
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     documents.append(f.read())
#             # Add vectorized documents to Qdrant collection
#             vectorstore.add_documents(documents, metadata={"company": company_name})
#             print(f"Added {len(documents)} documents for {company_name}")
#             company_vectors[company_name] = documents
#     return company_vectors

# Function to parse PDF and text documents and store them in Qdrant
def process_financial_documents(folder_path):
    company_vectors = {}
    for company_name in os.listdir(folder_path):
        print(f"Processing documents for {company_name}")
        company_folder = os.path.join(folder_path, company_name)
        print("Company folder", company_folder)
        if os.path.isdir(company_folder):
            documents = []
            for file in os.listdir(company_folder):
                file_path = os.path.join(company_folder, file)
                if file.endswith('.pdf'):
                    pdf_text = parse_pdf(file_path)
                    documents.append(pdf_text)
                elif file.endswith('.txt'):
                    # If it's a text file, read it directly
                    with open(file_path, 'r', encoding='utf-8') as f:
                        documents.append(f.read())
            print(f"Added {len(documents)} documents for {company_name}")
            # Add vectorized documents to Qdrant collection
            vectorstore.add_documents(documents, metadata={"company": company_name})
            company_vectors[company_name] = documents
    return company_vectors

# Helper function to parse a PDF and extract its text
def parse_pdf(file_path):
    pdf_text = ""
    try:
        # Open the PDF file using PyMuPDF (fitz)
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pdf_text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return pdf_text

# Scraping stock prices, volatility, and PE ratio from Yahoo Finance
def get_stock_data(company_name, num_months):

    print(f"Getting stock data for {company_name}")
    stock = yf.Ticker(company_name)
    benchmark = yf.Ticker("^GSPC")  # S&P 500 Index for beta calculation

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*num_months)

    stock_hist = stock.history(start=start_date, end=end_date)
    benchmark_hist = benchmark.history(start=start_date, end=end_date)

    # Calculate returns
    stock_returns = stock_hist['Close'].pct_change().dropna()
    benchmark_returns = benchmark_hist['Close'].pct_change().dropna()

    # Align dates between stock and benchmark returns
    stock_returns, benchmark_returns = stock_returns.align(benchmark_returns, join='inner')

    # Volatility calculation
    volatility = stock_returns.std() * np.sqrt(252)  # Annualized volatility

    # PE ratio
    pe_ratio = stock.info['trailingPE'] if 'trailingPE' in stock.info else None

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
    news_url = f"https://finance.yahoo.com/quote/{company_name}/news"
    news_content = requests.get(news_url).text
    soup = BeautifulSoup(news_content, 'html.parser')
    news_articles = soup.find_all('h3')
    news_texts = [article.get_text() for article in news_articles]

    return {
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

    total = len(sentiments)
    return (positive_sentiment - negative_sentiment) / total  # Net sentiment score

# Standard QA chain using LangChain RetrievalQAChain
def risk_analysis(company_name, question):
    retriever = vectorstore.as_retriever()

    # Use a simple RetrievalQAChain to retrieve relevant documents and generate the answer
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=azure_llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # Retrieve relevant company documents
    context = qa_chain.run(question)

    # Get stock data
    stock_data = get_stock_data(company_name, 6)  # Assume 6 months for example
    sentiment_score = get_sentiment(stock_data["news_texts"])

    # Final risk analysis
    risk = "High" if stock_data["volatility"] > 0.3 or sentiment_score < 0 else "Low"
    strategy = "Buy" if stock_data["pe_ratio"] < 15 and sentiment_score > 0 else "Hold/Sell"

    # TLM Trustworthiness Score (Hypothetical Implementation)
    trustworthiness_score = np.random.uniform(0.7, 0.9)  # Placeholder for TLM score, random for now

    # Print financial metrics and analysis
    print(f"Company: {company_name}")
    print(f"Volatility: {stock_data['volatility']:.2f}")
    print(f"PE Ratio: {stock_data['pe_ratio']}")
    print(f"Beta: {stock_data['beta']:.2f}")
    print(f"Alpha: {stock_data['alpha']:.4f}")
    print(f"R-squared: {stock_data['r_squared']:.2f}")
    print(f"Standard Deviation: {stock_data['std_dev']:.4f}")
    print(f"Sharpe Ratio: {stock_data['sharpe_ratio']:.2f}")
    print(f"Sentiment Score: {sentiment_score:.2f}")
    print(f"Risk: {risk}")
    print(f"Strategy: {strategy}")
    print(f"Trustworthiness Score: {trustworthiness_score:.2f}")

    # Return final analysis with retrieved documents
    return context

# Main function to process the folder of financial documents
def main():
    folder_path = "C:\\Users\\CQTF47\\Desktop\\IU Masters\\Thesis\\Data"
    num_months = 6  # Example input
    question = "What is the company's financial health?"

    # Process financial documents and store in vector DB (Qdrant)
    company_vectors = process_financial_documents(folder_path)

    for company_name in company_vectors.keys():
        # Perform risk analysis and trading strategy generation
        response = risk_analysis(company_name, question)
        print(f"Generated Response for {company_name}: {response}")

if __name__ == "__main__":
    main()