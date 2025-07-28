import requests
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import time

# Replace with your Together.ai API key
TOGETHER_API_KEY = "d169fdf80972ab1897c66ef02804c00ec9cbbcbf39dfdec497880edbeddaa6f8"

# User input
user_input = input("You: ")

# API request
headers = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "prompt": user_input,
    "max_tokens": 100
}

response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=data)
output = response.json()

# ✅ Fix: Correct access to AI response
ai_reply = output['choices'][0]['text'].strip()
print("\nAI says:", ai_reply)

# Text-to-speech
tts = gTTS(text=ai_reply, lang='en')
filename = f"together_reply_{int(time.time())}.mp3"
tts.save(filename)

# Play audio
audio = AudioSegment.from_mp3(filename)
play(audio)
