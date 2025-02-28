import os
import yfinance as yf
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
# from llama_index import 
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader, Settings
# from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
# from llama_index.query_engine import VectorIndexQueryEngine
from transformers import pipeline
import numpy as np
import statsmodels.api as sm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import logging
# import openai  # For GPT-4 API

logging.basicConfig(level=logging.INFO)
logging.info("Imported all modules")

RISK_FREE_RATE = 0.02  # Risk-free rate for Sharpe ratio
collection_name = "financial_documents"

logging.info("Initialized all variables")


qdrant_client = QdrantClient(path="financial_analysis")

if qdrant_client.collection_exists(collection_name="financial_documents"):
    qdrant_client.delete_collection(collection_name="financial_documents")

qdrant_client.recreate_collection(
    collection_name="financial_documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)  # Assuming vector size of 1536
)

logging.info("Created Qdrant collection")

# vector_store = QdrantVectorStore(
#     client=qdrant_client,
#     collection_name="financial_documents"
# )

OPENAI_API_KEY = "3a6b230b917b4893a150f0ad7fa126cf"
os.environ["AZURE_OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://cpe-clx-openai.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-05-15" 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

openai_mm_llm = AzureOpenAIMultiModal(
    engine="cpe-clx-gpt4o",
    api_version=os.environ["OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    model="gpt-4o-2024-05-13",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    max_new_tokens=1500,
    max_retries = 1
)

embed_model_openai = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    # deployment_name="cpe-clx-embedding",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"] ,
    azure_deployment="cpe-clx-embedding"
)

Settings.llm = openai_mm_llm
Settings.embed_model = embed_model_openai

logging.info("Initialized Azure OpenAI models")

# Updated function to parse and vectorize financial documents using SimpleDirectoryReader and LlamaIndex
def process_financial_documents(folder_path):

    global index
    company_vectors = {}

    # if qdrant_client.collection_exists(collection_name="financial_documents"):

    #     collection_info = qdrant_client.get_collection(collection_name)

    #     print(f"Number of vectors: {collection_info.vectors_count}")

    #     logging.info("Using Existing Qdrant collection")

    #     vector_store = QdrantVectorStore(
    #                                      client=qdrant_client,
    #                                      collection_name="financial_documents"
    #                                     )
    #     index = VectorStoreIndex.from_vector_store(vector_store)

    #     company_vectors['ADS.DE'] = index

    # else:

    for company_name in os.listdir(folder_path):

        company_folder = os.path.join(folder_path, company_name)

        print(f"Processing documents for {company_name}")

        if os.path.isdir(company_folder):
            # Use SimpleDirectoryReader to load documents
            documents = SimpleDirectoryReader(company_folder).load_data()

            vector_store = QdrantVectorStore(
                                            client=qdrant_client,
                                            collection_name="financial_documents"
                                            )

            logging.info("Initialized Qdrant vector store")

            # Create index for each company using vector store
            index = VectorStoreIndex.from_documents(documents=documents,vector_store=vector_store)
            
            # index.create_index()

            company_vectors[company_name] = index

    return company_vectors

# Scraping stock prices, volatility, and PE ratio from Yahoo Finance
def get_stock_data(company_name, num_months):
    stock = yf.Ticker(company_name)
    benchmark = yf.Ticker("^GSPC")  # S&P 500 Index for beta calculation

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * num_months)

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

# Function to query GPT-4 with financial data and question using Azure OpenAI
def query_gpt_4_and_tlm(company_name, stock_data, sentiment_score, risk, strategy, question):
    # Prepare context for Azure OpenAI
    context = f"""
    Company: {company_name}
    Volatility: {stock_data['volatility']:.2f}
    PE Ratio: {stock_data['pe_ratio']}
    Beta: {stock_data['beta']:.2f}
    Alpha: {stock_data['alpha']:.4f}
    R-squared: {stock_data['r_squared']:.2f}
    Standard Deviation: {stock_data['std_dev']:.4f}
    Sharpe Ratio: {stock_data['sharpe_ratio']:.2f}
    Sentiment Score: {sentiment_score:.2f}
    Risk: {risk}
    Strategy: {strategy}
    Question: {question}
    """

    # Query the Azure OpenAI model using LlamaIndex's AzureOpenAI class
    gpt_response = openai_mm_llm.predict(context)

    # Placeholder for trustworthiness score (this would be calculated separately)
    trustworthiness_score = np.random.uniform(0.7, 0.9)  # Placeholder for actual TLM logic

    # Print GPT-4 response and trustworthiness score
    print(f"--- GPT-4 Response for {company_name} ---")
    print(gpt_response)
    print(f"Trustworthiness Score: {trustworthiness_score:.2f}")
    
    return gpt_response, trustworthiness_score


# Standard QA function using LlamaIndex query engine
def risk_analysis(company_name, question, index):
    
    query_engine = index.as_retriever(similarity_top_k=5) #VectorIndexQueryEngine(index=index)

    # Retrieve relevant company documents and generate the answer
    context = query_engine.retrieve(question)

    # Get stock data
    stock_data = get_stock_data(company_name, 6)  # Assume 6 months for example
    sentiment_score = get_sentiment(stock_data["news_texts"])

    # Final risk analysis
    risk = "High" if stock_data["volatility"] > 0.3 or sentiment_score < 0 else "Low"
    strategy = "Buy" if stock_data["pe_ratio"] < 15 and sentiment_score > 0 else "Hold/Sell"

    # Query GPT-4 via Azure OpenAI and get trustworthiness score
    gpt_response, trustworthiness_score = query_gpt_4_and_tlm(
        company_name, stock_data, sentiment_score, risk, strategy, question
    )

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
        response = risk_analysis(company_name, question, index)
        print(f"Generated Response for {company_name}: {response}")

if __name__ == "__main__":
    main()



































