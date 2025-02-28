from fastapi import FastAPI, Depends, HTTPException
from services.stock_service import StockService
from services.sentiment_service import SentimentService
from services.risk_analysis_service import RiskAnalysisService
from services.gpt_service import GPTService
from pydantic import BaseModel
import logging
import nest_asyncio
nest_asyncio.apply()

app = FastAPI()

# Request and Response Models for FastAPI
class RiskAnalysisRequest(BaseModel):
    company_name: str
    question: str

class RiskAnalysisResponse(BaseModel):
    gpt_response: str
    trustworthiness_score: float

# Dependency Injection of services
def get_stock_service():
    return StockService()

def get_sentiment_service():
    return SentimentService()

def get_risk_analysis_service():
    return RiskAnalysisService()

def get_gpt_service():
    return GPTService()

@app.get("/")
async def root():
    return {"message": "Financial Risk Analysis API"}

@app.post("/analyze", response_model=RiskAnalysisResponse)
async def analyze_stock_risk(
    request: RiskAnalysisRequest,
    stock_service: StockService = Depends(get_stock_service),
    sentiment_service: SentimentService = Depends(get_sentiment_service),
    risk_analysis_service: RiskAnalysisService = Depends(get_risk_analysis_service),
    gpt_service: GPTService = Depends(get_gpt_service)
):
    try:
        # Fetch stock data
        stock_data = stock_service.get_stock_data(request.company_name, 6)

        # Perform sentiment analysis
        sentiment_score = sentiment_service.analyze_sentiment(stock_data["news_texts"])

        # Perform risk analysis
        risk, strategy = risk_analysis_service.analyze_risk(stock_data, sentiment_score)

        # Query GPT-4 and get trustworthiness score
        gpt_response, trustworthiness_score = gpt_service.query_gpt_4_and_tlm(
            request.company_name, stock_data, sentiment_score, risk, strategy, request.question
        )

        return RiskAnalysisResponse(
            gpt_response=gpt_response,
            trustworthiness_score=trustworthiness_score['trustworthiness_score']
        )
    except Exception as e:
        logging.error(f"Error occurred during analysis: {e}")
        raise HTTPException(status_code=500, detail="Error processing the request")

