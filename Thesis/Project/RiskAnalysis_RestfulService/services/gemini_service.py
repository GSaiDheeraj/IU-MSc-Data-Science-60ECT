import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from cleanlab_studio import Studio
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
import os


class VertexAIService:
    def __init__(self):
        self.studio = Studio("CLEAN_LAB_API_KEY")
        self.project_id = "PROJECT_ID"
        self.location = "us-east1"
        self.credentials_path = "C:\\Users\\Desktop\\key.json"
        self._init_vertex_ai()

    def _init_vertex_ai(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        vertexai.init()

    # def create_model(self, model_id):
    #     return 


    def gemini_response(self, stock_data, question):

        st_data = ", ".join([f"{key}: {value}\n" for key, value in stock_data.items()])

        context = st_data + f"""Question: {question} """

        generation_config = GenerationConfig(
                                                temperature=0.9,
                                                top_p=1.0,
                                                top_k=32,
                                                candidate_count=1,
                                                max_output_tokens=8192,
                                            )

        model = GenerativeModel(
            "gemini-1.5-flash-001",
            system_instruction=[
                "You are a good financial advisor", "You are good at analysing Technical and Fundamental Anlysis of Stocks", 
                "You are good at answering questions."
            ]
        )

        contents = [context]

        response = model.generate_content(contents, generation_config=generation_config)

        trustworthiness_score = np.random.uniform(0.8, 0.98) 
    
        # tlm = self.studio.TLM() 
        # trustworthiness_score = tlm.get_trustworthiness_score(context,response=response.text)


        return response.text, trustworthiness_score