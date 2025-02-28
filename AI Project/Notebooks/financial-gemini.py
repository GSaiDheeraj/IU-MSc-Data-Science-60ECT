from flask import Flask, jsonify, request
import requests

# Dependency injection for configuration management (Open/Closed Principle)
class Config:
    API_KEY = 'm,(w(7-CzS-kbVY8u3AF.JYeAG-7PLyuyP6.x@zj'
    CONTENT_TYPE = 'application/json'
    BASE_URL = 'https://genai-service.stage.commandcentral.com/app-gateway/api/v2/chat'

# Single Responsibility Principle: Handle the external API requests separately
class StockPredictionService:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def generate_prediction(self, data):
        headers = {
            'x-msi-genai-api-key': self.api_key,
            'Content-Type': Config.CONTENT_TYPE
        }
        response = requests.post(self.base_url, headers=headers, json=data)
        
        # Handle API response and return the relevant key
        if response.status_code == 200:
            return response.json().get('msg', 'No message found in response')
        else:
            return f"Error: {response.status_code}, {response.text}"

# Flask app definition (Adheres to the Dependency Inversion Principle)
app = Flask(__name__)

# Dependency injection of the StockPredictionService into the Flask app (Inversion of Control)
service = StockPredictionService(api_key=Config.API_KEY, base_url=Config.BASE_URL)

# Open/Closed Principle: The generate-response endpoint is flexible, allowing for different prompts without modifying the core logic
@app.route('/generate-response', methods=['POST'])
def generate_response():
    # Single Responsibility Principle: Process request inputs
    request_data = request.get_json()
    
    if not request_data or 'prompt' not in request_data:
        return jsonify({'error': 'Invalid input, prompt is required'}), 400

    # Encapsulate the data creation logic
    data = {
        'userId': 'cqtf47@motorolasolutions.com',
        'model': 'VertexGemini',
        'prompt': request_data['prompt'],
        'datastoreId': '362a5d9c-65f8-43df-a751-2057708501ca'
    }

    # Call the service to generate the prediction
    prediction_message = service.generate_prediction(data)
    
    return jsonify({'msg': prediction_message})

# Liskov Substitution: The Flask app should behave the same with or without dependency changes
if __name__ == '__main__':
    app.run(debug=True)
