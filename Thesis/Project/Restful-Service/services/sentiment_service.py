from transformers import pipeline

class SentimentService:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")

    def analyze_sentiment(self, news_texts: list):
        # print('sentiment_pipeline', news_texts)
        sentiments = self.sentiment_pipeline(news_texts)
        positive_sentiment = sum([1 for s in sentiments if s['label'] == 'POSITIVE'])
        negative_sentiment = sum([1 for s in sentiments if s['label'] == 'NEGATIVE'])
        total = len(sentiments)
        return (positive_sentiment - negative_sentiment) / total  # Net sentiment score
