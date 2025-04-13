import ollama
import transformers
from openai import OpenAI
import os 
import dotenv
dotenv.load_dotenv()

class YogobellaMLLMix: 

    def ollama_model(self, model_name="deepseek-r1", messages= [{'role': 'user', 'content': 'What is the capital of Slovenia?'}]):
        response = ollama.chat(
            model=model_name,
            messages=messages
        )
        return response['message']['content']

    def gams(self, messages=[{"role": "user", "content": "kaj je dežela?"}]):

        model_id = "cjvt/OPT_GaMS-1B-Chat"
        pipeline = transformers.pipeline("text-generation", model=model_id, device_map="auto")
        response = pipeline(messages, max_length=100, )
        print("Model's response:", response[0]["generated_text"][1]["content"])

    def pay_to_win_AIM(self, model_name="gpt-4o", messages=[{"role": "user", "content": "Write a one-sentence story about numbers."}]):
        client = OpenAI(
            base_url="https://api.aimlapi.com/v1",
            api_key=os.getenv("AIM_APY_KEY"),    
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        
        return response.choices[0].message.content