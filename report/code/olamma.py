import ollama

response = ollama.chat(
    model='deepseek-r1',  # or 'mistral', 'gemma', etc.
    messages=[
        {'role': 'user', 'content': 'What is the capital of Slovenia?'}
    ]
)

print(response['message']['content'])
