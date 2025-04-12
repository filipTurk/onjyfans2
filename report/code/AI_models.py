import olamma
import transformers
from openai import OpenAI
import os 
import dotenv
dotenv.load_dotenv()

class YogobellaMLLMix: 
    
    def __init__(self, model_name, model_path, tokenizer_path):
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    def ollama(self, model_name="deepseek-r1", messages= [{'role': 'user', 'content': 'What is the capital of Slovenia?'}]):
        response = olamma.chat(
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