class RiskAnalysisService:
    def analyze_risk(self, stock_data, sentiment_score):
        risk = "High" if stock_data["volatility"] > 0.3 or sentiment_score < 0 else "Low"
        strategy = "Buy" if stock_data["pe_ratio"] < 15 and sentiment_score > 0 else "Hold/Sell"
        return risk, strategy
