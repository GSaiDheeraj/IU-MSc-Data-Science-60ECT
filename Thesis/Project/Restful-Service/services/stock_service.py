import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
from bs4 import BeautifulSoup
import requests

class StockService:
    RISK_FREE_RATE = 0.02  # Risk-free rate for Sharpe ratio

    def get_stock_data(self, company_name: str, num_months: int):
        stock = yf.Ticker(company_name)

        # print("stock info", stock.info)

        if company_name.split(".")[-1] == "NS":
           benchmark = yf.Ticker("^NSEI")  # Example benchmark for beta calculation
        elif company_name.split(".")[-1] == "BO":
           benchmark = yf.Ticker("^BSESN")
        elif company_name.split(".")[-1] == "AX":
            benchmark = yf.Ticker("^AXJO")
        elif company_name.split(".")[-1] == "L":
            benchmark = yf.Ticker("^FTSE")
        elif company_name.split(".")[-1] == "DE":
            benchmark = yf.Ticker("^GDAXI")
        else:
            benchmark = yf.Ticker("^IXIC")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * num_months)
        stock_hist = stock.history(start=start_date, end=end_date)
        benchmark_hist = benchmark.history(start=start_date, end=end_date)

        stock_returns = stock_hist['Close'].pct_change().dropna()
        benchmark_returns = benchmark_hist['Close'].pct_change().dropna()
        stock_returns, benchmark_returns = stock_returns.align(benchmark_returns, join='inner')

        volatility = stock_returns.std() * np.sqrt(252)
        pe_ratio = stock.info.get('trailingPE', None)

        X = sm.add_constant(benchmark_returns)
        model = sm.OLS(stock_returns, X).fit()
        beta = model.params[1]
        alpha = model.params[0]
        r_squared = model.rsquared
        std_dev = stock_returns.std()
        mean_return = stock_returns.mean() * 252
        sharpe_ratio = (mean_return - self.RISK_FREE_RATE) / std_dev
        news_texts = [i['content']['title'] for i in stock.news] #self.scrape_news(company_name)

        # print("news", news_texts)

        # Keys to extract
        income_extract = list(stock.income_stmt.index)
        balance_extract = list(stock.balance_sheet.index)
        cashflow_extract = list(stock.cashflow.index)

        # Dictionary to store extracted data
        extracted_data = {}

        # Extract rows for each key
        for key in income_extract:
            if key in stock.income_stmt.index:
                extracted_data[key] = stock.income_stmt.loc[key]
        for key in balance_extract:
            if key in stock.balance_sheet.index:
                extracted_data[key] = stock.balance_sheet.loc[key]
        for key in cashflow_extract:
            if key in stock.cashflow.index:
                extracted_data[key] = stock.cashflow.loc[key]

        # print("extracted_data",extracted_data)
        # print("volatility", volatility)
        # print("pe_ratio", pe_ratio)
        # print("beta", beta)
        # print("alpha", alpha)
        # print("r_squared", r_squared)
        # print("std_dev", std_dev)
        # print("sharpe_ratio", sharpe_ratio)
        # print("news_texts", news_texts)
        # print("eps_trend_90day", stock.eps_trend['90daysAgo'])
        # print("eps_trend_60day", stock.eps_trend['60daysAgo'])
        # print("eps_trend_30day", stock.eps_trend['30daysAgo'])
        # print("eps_trend_current", stock.eps_trend['current'])
        # print("growth_estimate", stock.growth_estimates['stock'])
        # print("stock_price open history", stock_hist['Open'])
        # print("stock_price close history", stock_hist['Close'])

        return_dict = {
            "volatility": volatility,
            "pe_ratio": pe_ratio,
            "beta": beta,
            "alpha": alpha,
            "r_squared": r_squared,
            "std_dev": std_dev,
            "sharpe_ratio": sharpe_ratio,
            "news_texts": news_texts,
            "eps_trend_90day": stock.eps_trend['90daysAgo'],
            "eps_trend_60day": stock.eps_trend['60daysAgo'],
            "eps_trend_30day": stock.eps_trend['30daysAgo'],
            "eps_trend_current": stock.eps_trend['current'],
            "growth_estimate": stock.growth_estimates['stockTrend'],
            "stock_price open history": stock_hist['Open'],
            "stock_price close history": stock_hist['Close'],
            "stock returns": stock_returns,
            "benchmark returns": benchmark_returns,
            "Current Analyst Price Target": stock.analyst_price_targets['current'],
            "Max Analyst Price Target": stock.analyst_price_targets['high'],
            "Min Analyst Price Target": stock.analyst_price_targets['low'],
            "Analyst Earnigns Estimate": stock.earnings_estimate['avg']
        }
        return_dict.update(extracted_data)

        # print(return_dict)

        return return_dict

        # return {
        #     "volatility": volatility,
        #     "pe_ratio": pe_ratio,
        #     "beta": beta,
        #     "alpha": alpha,
        #     "r_squared": r_squared,
        #     "std_dev": std_dev,
        #     "sharpe_ratio": sharpe_ratio,
        #     "news_texts": news_texts,
        #     "eps_trend_90day": stock.eps_trend['90daysAgo'],
        #     "eps_trend_60day": stock.eps_trend['60daysAgo'],
        #     "eps_trend_30day": stock.eps_trend['30daysAgo'],
        #     "eps_trend_current": stock.eps_trend['current'],
        #     "growth_estimate": stock.growth_estimates['stock'],
        #     "stock_price open history": stock_hist['Open'],
        #     "stock_price close history": stock_hist['Close'],
        #     "stock returns": stock_returns,
        #     "benchmark returns": benchmark_returns,
        #     # "EBITDA": stock.income_stmt.iloc[8],
        #     # "EBIT": stock.income_stmt.iloc[9],
        #     # "Net Income": stock.income_stmt.iloc[24],
        #     # "Net Debt": stock.balance_sheet.iloc[2],
        #     # "Total Debt": stock.balance_sheet.iloc[3],
        #     # "Total Assests": stock.balance_sheet.iloc[43],
        #     # "Payables":stock.balance_sheet.iloc[38],
        #     # "Gross Accounts Receivable": stock.balance_sheet.iloc[79],
        #     # "Change in Payables": stock.cashflow.iloc[40],
        #     # "Free Cash Flow": stock.cashflow.iloc[0],
        #     "Current Analyst Price Target": stock.analyst_price_targets['current'],
        #     "Max Analyst Price Target": stock.analyst_price_targets['high'],
        #     "Min Analyst Price Target": stock.analyst_price_targets['low'],
        #     "Analyst Earnigns Estimate": stock.earnings_estimate['avg']
        # }

    def scrape_news(self, company_name: str):
        from bs4 import BeautifulSoup
        import requests

        news_url = f"https://finance.yahoo.com/lookup/?s={company_name}"
        news_content = requests.get(news_url)
        soup = BeautifulSoup(news_content.content, 'html.parser')
        news_articles = [h3.get_text() for h3 in soup.find_all('h3')]

        return [heading for heading in news_articles if len(heading.split()) > 4]
