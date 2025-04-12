import requests

response = requests.post(
    'http://localhost:11434/api/chat',
    json={
        'model': 'llama2',
        'messages': [
            {'role': 'user', 'content': 'Tell me a joke.'}
        ]
    }
)

print(response.json()['message']['content'])
