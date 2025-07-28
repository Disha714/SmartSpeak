from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
from gtts import gTTS
from datetime import datetime
from pydub import AudioSegment
from textblob import TextBlob
from langdetect import detect
from transformers import pipeline
import os
import time
import tempfile
import requests

app = Flask(__name__)
app.secret_key = "smartspeak_secret_key"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
app.config['UPLOAD_FOLDER'] = 'static'

# LLM API key
TOGETHER_API_KEY = "d169fdf80972ab1897c66ef02804c00ec9cbbcbf39dfdec497880edbeddaa6f8"

# HuggingFace summarization
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# ========== Helpers ==========

def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = f"output_{int(time.time())}.mp3"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    tts.save(path)
    return filename

# ========== Routes ==========

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tts", methods=["GET", "POST"])
def tts():
    audio_file = None
    if request.method == "POST":
        text = request.form["text"]
        voice = request.form.get("voice", "en")
        lang_map = {
            "male": "en",
            "female": "en",
            "india": "en-in",
            "us": "en",
            "uk": "en-uk"
        }
        lang = lang_map.get(voice, "en")
        audio_file = speak(text, lang)
    return render_template("tts.html", audio_file=audio_file)

@app.route("/sentiment", methods=["GET", "POST"])
def sentiment():
    if request.method == "POST":
        text = request.form["text"]
        polarity = TextBlob(text).sentiment.polarity
        tone = "Positive" if polarity > 0.4 else "Negative" if polarity < -0.4 else "Neutral"
        audio_file = speak(f"The sentiment is {tone}")
        return render_template("sentiment.html", sentiment=tone, audio_file=audio_file)
    return render_template("sentiment.html")

@app.route("/stt", methods=["GET", "POST"])
def stt():
    if request.method == "POST":
        audio = request.files["audio"]
        path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
        audio.save(path)
        # Simulated STT result (actual STT requires a model like whisper)
        result = "This is a simulated transcript."
        return render_template("stt.html", transcript=result)
    return render_template("stt.html")

@app.route("/summarize", methods=["GET", "POST"])
def summarize():
    if request.method == "POST":
        audio = request.files["audio"]
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
        audio.save(audio_path)

        # Fake transcription for demo purposes
        fake_text = "SmartSpeak is an AI voice assistant project created to demonstrate audio processing."
        summary = summarizer(fake_text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
        audio_file = speak(summary)
        return render_template("summarize.html", summary=summary, audio_file=audio_file)
    return render_template("summarize.html")

@app.route("/translate", methods=["GET", "POST"])
def translate():
    translated_text = None
    if request.method == "POST":
        text = request.form["text"]
        target_lang = request.form["target_lang"]
        # Simulated translation
        translated_text = f"(Translated to {target_lang}) " + text[::-1]
        audio_file = speak(translated_text, lang=target_lang)
        return render_template("translate.html", translated_text=translated_text, audio_file=audio_file)
    return render_template("translate.html")

@app.route("/llm", methods=["GET", "POST"])
def ask_llm():
    if "chat_history" not in session:
        session["chat_history"] = []

    reply = None
    audio_file = None

    if request.method == "POST":
        prompt = request.form["prompt"]
        history = session["chat_history"]

        full_prompt = "\n".join([
            f"User: {entry[0]}\nAI: {entry[1]}" if isinstance(entry, tuple)
            else f"User: {entry['question']}\nAI: {entry['answer']}"
            for entry in history
            ])
        full_prompt += f"\nUser: {prompt}\nAI:"

        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "prompt": full_prompt,
            "max_tokens": 100
        }

        try:
            response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=data)
            result = response.json()
            reply = result["choices"][0]["text"].strip() if "choices" in result else "No reply received."

            timestamp = datetime.now().strftime("%I:%M %p")
            session["chat_history"].append({
    "question": prompt,
    "answer": reply,
    "timestamp": datetime.now().strftime("%H:%M:%S")
})

            session.modified = True
            audio_file = speak(reply)
        except Exception as e:
            reply = f"Error: {e}"

    return render_template("llm_chat.html", history=session["chat_history"], audio_file=audio_file)

@app.route("/reset_chat")
def reset_chat():
    session.pop("chat_history", None)
    return redirect(url_for("ask_llm"))

# ========== Run Server ==========
if __name__ == "__main__":
    app.run(debug=True)
