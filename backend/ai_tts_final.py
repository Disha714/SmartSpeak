import os
import time
import requests
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from textblob import TextBlob
from langdetect import detect
from transformers import pipeline
import speech_recognition as sr

# Initialize models once
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
recognizer = sr.Recognizer()

# Replace with your actual API key
TOGETHER_API_KEY = "d169fdf80972ab1897c66ef02804c00ec9cbbcbf39dfdec497880edbeddaa6f8"

def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = f"output_{int(time.time())}.mp3"
    tts.save(filename)
    audio = AudioSegment.from_mp3(filename)
    play(audio)

def get_input():
    print("\nChoose input method:")
    print("1. Type")
    print("2. Speak")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        return input("Enter your text: ")
    elif choice == '2':
        try:
            with sr.Microphone() as source:
                print("🎤 Speak now...")
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
        except Exception as e:
            print("STT failed:", e)
            return ""
    else:
        print("Invalid choice. Defaulting to typed input.")
        return input("Enter your text: ")

def summarization_mode():
    print("\n📝 Summarization Mode")
    text = get_input()
    if not text:
        return
    if len(text.split()) > 500:
        print("Text too long. Limit to ~500 words.")
        return
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    print("Summary:\n", summary)
    speak(summary)

def language_detection_mode():
    print("\n🌐 Language Detection Mode")
    text = get_input()
    if not text:
        return
    lang = detect(text)
    print(f"Detected language: {lang}")
    speak(text, lang=lang)

def sentiment_mode():
    print("\n🧠 Sentiment Analysis Mode")
    text = get_input()
    if not text:
        return
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0.4:
        tone = "positive"
    elif sentiment < -0.4:
        tone = "negative"
    else:
        tone = "neutral"
    print(f"Detected sentiment: {tone} (score: {sentiment})")
    speak(text)

def ask_llm_mode():
    print("\n🤖 Ask AI (Together.ai)")
    prompt = get_input()
    if not prompt:
        return
    headers = {
        "Authorization": f"Bearer {d169fdf80972ab1897c66ef02804c00ec9cbbcbf39dfdec497880edbeddaa6f8}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "prompt": prompt,
        "max_tokens": 100
    }
    try:
        response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=data)
        result = response.json()
        ai_reply = result['choices'][0]['text'].strip()
        print("AI says:\n", ai_reply)
        speak(ai_reply)
    except Exception as e:
        print("LLM request failed:", e)

def main_menu():
    while True:
        print("\n" + "="*50)
        print("🎙️ AI-Powered TTS Assistant - Final Version")
        print("1. Summarize Text")
        print("2. Detect Language & Speak")
        print("3. Sentiment-Based Speaking")
        print("4. Ask an LLM (Together.ai)")
        print("5. Exit")
        print("="*50)
        choice = input("Choose an option (1-5): ")

        if choice == '1':
            summarization_mode()
        elif choice == '2':
            language_detection_mode()
        elif choice == '3':
            sentiment_mode()
        elif choice == '4':
            ask_llm_mode()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main_menu()
