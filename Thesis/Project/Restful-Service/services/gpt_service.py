import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from cleanlab_studio import Studio
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
# from ragrank import evaluate
# from ragrank.dataset import from_dict
import os

class GPTService:
    def __init__(self):
        self.studio = Studio("CLEAN_LAB_API_KEY") #Studio("94ae2b40d9414d4b873b1a94d3da5999")
        self.llm = AzureChatOpenAI(
            model="gpt4o",
            azure_deployment="gpt4o",
            api_key="API_KEY",
            azure_endpoint="azure.com",
            api_version="2023-05-15"
        )
        self.embedding_model = AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key="API_KEY",
            azure_endpoint="azure.com/",
            api_version="2023-05-15",
            azure_deployment="embedding"
        )
        

    def query_gpt_4_and_tlm(self, stock_data, question):
        
        st_data = ", ".join([f"{key}: {value}\n" for key, value in stock_data.items()])

        context = st_data

        import json
        with open('data.json', 'w') as f:
            json.dump(context, f)

        prompt_template = PromptTemplate(
                                            input_variables=['question'], 
                                            template=context + "\nQuestion: {question}"
                                        )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt_template)
        answer = llm_chain.run(question)

        # Trustworthiness score from CleanLab Studio
        tlm = self.studio.TLM() 
        trustworthiness_score = tlm.get_trustworthiness_score(context,response=answer) #np.random.uniform(0.8, 1.0) #tlm.get_trustworthiness_score(context,response=answer)
        # print(trustworthiness_score)

        #ragrank
        # eval = from_dict({
        #                     "question": question,
        #                     "context": [context],
        #                     "response": answer,
        #                 })
        
        # result = evaluate(eval)
        # print(result)

        return answer, trustworthiness_score
