import gradio as gr
from gtts import gTTS
from pydub import AudioSegment
import time
import os
import requests
from textblob import TextBlob
from langdetect import detect
from transformers import pipeline
import tempfile

# Init models
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
TOGETHER_API_KEY = "PASTE_YOUR_TOGETHER_API_KEY_HERE"

# TTS helper
def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    temp_path = os.path.join(tempfile.gettempdir(), f"out_{int(time.time())}.mp3")
    tts.save(temp_path)
    return temp_path

# Summarize
def summarize_text(text):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    mp3_path = speak(summary)
    return summary, mp3_path

# Sentiment
def sentiment_tts(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    tone = "positive" if polarity > 0.4 else "negative" if polarity < -0.4 else "neutral"
    mp3_path = speak(text)
    return f"Sentiment: {tone}", mp3_path

# Language detection
def detect_language(text):
    lang = detect(text)
    mp3_path = speak(text, lang)
    return f"Language: {lang}", mp3_path

# LLM using Together.ai
def ask_llm(prompt):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
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

        if "choices" not in result:
            return f"API Error: {result}", None

        reply = result['choices'][0]['text'].strip()
        mp3_path = speak(reply)
        return reply, mp3_path

    except Exception as e:
        return f"Error: {e}", None


# UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ AI Voice Assistant - Web Version")

    with gr.Tab("📝 Summarize"):
        input_text = gr.Textbox(label="Enter text to summarize")
        summary_output = gr.Textbox(label="Summary")
        summary_audio = gr.Audio(label="Speech Output", type="filepath")
        summarize_btn = gr.Button("Summarize & Speak")
        summarize_btn.click(fn=summarize_text, inputs=input_text, outputs=[summary_output, summary_audio])

    with gr.Tab("🧠 Sentiment"):
        senti_text = gr.Textbox(label="Enter text")
        senti_output = gr.Textbox(label="Detected Sentiment")
        senti_audio = gr.Audio(label="Speech Output", type="filepath")
        senti_btn = gr.Button("Speak Sentiment")
        senti_btn.click(fn=sentiment_tts, inputs=senti_text, outputs=[senti_output, senti_audio])

    with gr.Tab("🌐 Language Detection"):
        lang_text = gr.Textbox(label="Enter text in any language")
        lang_output = gr.Textbox(label="Detected Language")
        lang_audio = gr.Audio(label="Speech Output", type="filepath")
        lang_btn = gr.Button("Detect & Speak")
        lang_btn.click(fn=detect_language, inputs=lang_text, outputs=[lang_output, lang_audio])

    with gr.Tab("🤖 Ask AI"):
        llm_input = gr.Textbox(label="Ask a question")
        llm_output = gr.Textbox(label="AI Response")
        llm_audio = gr.Audio(label="Speech Output", type="filepath")
        llm_btn = gr.Button("Get AI Answer & Speak")
        llm_btn.click(fn=ask_llm, inputs=llm_input, outputs=[llm_output, llm_audio])

demo.launch()
