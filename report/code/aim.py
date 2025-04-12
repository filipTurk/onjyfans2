from openai import OpenAI
import os 
import dotenv
dotenv.load_dotenv()


client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=os.getenv("AIM_APY_KEY"),    
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a one-sentence story about numbers."}]
)

print(response.choices[0].message.content)