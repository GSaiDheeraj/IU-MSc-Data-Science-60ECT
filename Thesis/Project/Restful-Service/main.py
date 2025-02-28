from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from services.stock_service import StockService
from services.sentiment_service import SentimentService
# from services.risk_analysis_service import RiskAnalysisService
from services.gpt_service import GPTService
from services.gemini_service import VertexAIService
# from deepeval.metrics import TrustworthinessMeteric
import logging
import time
import nest_asyncio
nest_asyncio.apply()

# Your existing imports and financial analysis functions go here...

app = FastAPI()

# Mount static files for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates for Jinja2
templates = Jinja2Templates(directory="templates")

# Request model for financial analysis
class AnalyzeRequest(BaseModel):
    company_name: str
    question: str

class ScrapperRequest(BaseModel):
    company_name: str

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

# def get_risk_analysis_service():
#     return RiskAnalysisService()

def get_gpt_service():
    return GPTService()

def get_vertex_service():
    return VertexAIService()

# Render the UI page
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stockinformation", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("indexscrapper.html", {"request": request})

# API endpoint to handle stock analysis requests
@app.post("/analyze")
async def analyze_stock(request: AnalyzeRequest,
                        stock_service: StockService = Depends(get_stock_service),
                        sentiment_service: SentimentService = Depends(get_sentiment_service),
                        # risk_analysis_service: RiskAnalysisService = Depends(get_risk_analysis_service),
                        gpt_service: GPTService = Depends(get_gpt_service),
                        gemini_service: VertexAIService = Depends(get_vertex_service)):
    try:
        st_time = time.time()
        stock_data = stock_service.get_stock_data(request.company_name, 6)
        print('Scraper Service Time:', time.time() - st_time)

        # Perform sentiment analysis
        st_time = time.time()
        sentiment_score = sentiment_service.analyze_sentiment(stock_data["news_texts"])
        print('Sentiment Service Time:', time.time() - st_time)

        # Perform risk analysis
        # risk, strategy = risk_analysis_service.analyze_risk(stock_data, sentiment_score)

        # Query GPT-4 and get trustworthiness score
        st_time = time.time()
        gpt_response, gpt_trustworthiness_score = gpt_service.query_gpt_4_and_tlm(
             stock_data, request.question
        )
        print('GPT Response Time:', time.time() - st_time)
        # print(gpt_response, gpt_trustworthiness_score)

        gemini_st_time = time.time()
        gemini_response, gemini_trustworthiness_score = gemini_service.gemini_response(
             stock_data, request.question
        )
        print('GPT Response Time:', time.time() - gemini_st_time)
        print("LLM Service Time:", time.time() - st_time)

        # Call your financial analysis function with the provided data
        # response, trustworthiness_score = risk_analysis(request.company_name, request.question)

        return JSONResponse(content={
            "gpt_response": gpt_response,
            "gpt_trustworthiness_score": gpt_trustworthiness_score['trustworthiness_score'],
            "gemini_response": gemini_response,
            "gemini_trustworthiness_score": gemini_trustworthiness_score['trustworthiness_score']
        })
    except Exception as e:
        logging.error(f"Error analyzing stock: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

@app.post("/webscrapper")
async def webscrapper(request: ScrapperRequest,
                        stock_service: StockService = Depends(get_stock_service)):
    try:
        st_time = time.time()
        stock_data = stock_service.get_stock_data(request.company_name, 6)
        print('Scraper Service Time:', time.time() - st_time)

        return JSONResponse(content={
            "data": str(stock_data),
        })
    except Exception as e:
        logging.error(f"Error analyzing stock: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)