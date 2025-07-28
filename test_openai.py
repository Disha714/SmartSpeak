import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # use environment variable instead


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

print(response['choices'][0]['message']['content'])
