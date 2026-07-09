from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
from gtts import gTTS
from datetime import datetime
from dotenv import load_dotenv
import os
import time
import requests

import stt
import translator
import summarize
import sentiment_tts
import rag_chat
import image_caption

# templates/ and static/ live one level up from backend/, not inside it --
# Flask's default (template_folder="templates" relative to this file's
# directory) would silently fail to find them, so point at them explicitly.
app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "smartspeak_secret_key")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
# Use Flask's resolved static_folder (absolute path) rather than the bare
# string "static" -- since the app is run from backend/, a relative "static"
# would resolve to backend/static (which doesn't exist) instead of the
# actual static/ folder one level up.
app.config["UPLOAD_FOLDER"] = app.static_folder

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ========== Helpers ==========


def speak(text, lang="en"):
    """Cloud TTS via gTTS -- used everywhere except /sentiment, which needs
    local prosody control (rate/volume) that gTTS doesn't expose."""
    tts = gTTS(text=text, lang=lang)
    filename = f"output_{int(time.time())}.mp3"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    tts.save(path)
    return filename


# ========== Routes ==========


@app.route("/")
def home():
    return render_template("index.html", active="home")


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
            "uk": "en-uk",
        }
        lang = lang_map.get(voice, "en")
        audio_file = speak(text, lang)
    return render_template("tts.html", audio_file=audio_file, active="tts")


@app.route("/sentiment", methods=["GET", "POST"])
def sentiment():
    if request.method == "POST":
        text = request.form["text"]
        tone, polarity = sentiment_tts.analyze_sentiment(text)
        # Speak the ORIGINAL text with prosody shaped by its own sentiment,
        # instead of just reading out the tone label as a flat sentence.
        audio_file = sentiment_tts.speak_with_emotion(
            text, polarity, output_dir=app.config["UPLOAD_FOLDER"]
        )
        return render_template(
            "sentiment.html", sentiment=tone, polarity=round(polarity, 2), audio_file=audio_file, active="sentiment"
        )
    return render_template("sentiment.html", active="sentiment")


@app.route("/stt", methods=["GET", "POST"])
def speech_to_text_route():
    if request.method == "POST":
        audio = request.files["audio"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], audio.filename)
        audio.save(path)
        result = stt.speech_to_text(path)
        return render_template("stt.html", transcript=result, active="stt")
    return render_template("stt.html", active="stt")


@app.route("/summarize", methods=["GET", "POST"])
def summarize_route():
    if request.method == "POST":
        audio = request.files["audio"]
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio.filename)
        audio.save(audio_path)

        # Actually transcribe the uploaded audio instead of summarizing a
        # hardcoded placeholder string regardless of what was uploaded.
        transcript = stt.speech_to_text(audio_path)
        summary = summarize.summarize_text(transcript)
        audio_file = speak(summary)
        return render_template(
            "summarize.html", transcript=transcript, summary=summary, audio_file=audio_file, active="summarize"
        )
    return render_template("summarize.html", active="summarize")


@app.route("/translate", methods=["GET", "POST"])
def translate():
    translated_text = None
    if request.method == "POST":
        text = request.form["text"]
        target_lang = request.form["target_lang"]
        translated_text = translator.translate_text(text, target_lang)
        audio_file = speak(translated_text, lang=target_lang)
        return render_template(
            "translate.html", translated_text=translated_text, audio_file=audio_file, active="translate"
        )
    return render_template("translate.html", active="translate")


@app.route("/image", methods=["GET", "POST"])
def image_to_speech():
    """New: upload an image, get a spoken caption. Extends the project from
    pure text/audio I/O into a multimodal (vision + speech) tool."""
    if request.method == "POST":
        image = request.files["image"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
        image.save(path)
        caption = image_caption.caption_image(path)
        audio_file = speak(caption)
        return render_template("image.html", caption=caption, audio_file=audio_file, active="image")
    return render_template("image.html", active="image")


@app.route("/llm", methods=["GET", "POST"])
def ask_llm():
    if "chat_history" not in session:
        session["chat_history"] = []

    audio_file = None

    if request.method == "POST":
        prompt = request.form["prompt"]
        history = session["chat_history"]

        # RAG: retrieve relevant past turns instead of stuffing the entire
        # unbounded history into every request.
        messages = rag_chat.build_messages(prompt, history)

        if not GROQ_API_KEY:
            reply = "Error: GROQ_API_KEY is not set. Add it to your .env file."
        else:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            data = {
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 300,
            }
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                reply = (
                    result["choices"][0]["message"]["content"].strip()
                    if "choices" in result
                    else f"No reply received: {result}"
                )
            except requests.exceptions.RequestException as e:
                reply = f"Error reaching the LLM API: {e}"

        session["chat_history"].append(
            {
                "question": prompt,
                "answer": reply,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )
        session.modified = True

        # Index this turn for future retrieval. Skipped silently if it
        # errored above -- no point embedding an error message as context.
        if not reply.startswith("Error"):
            try:
                rag_chat.add_turn(prompt, reply)
            except Exception:
                pass  # retrieval indexing is a nice-to-have, not critical path

        audio_file = speak(reply)

    return render_template("llm_chat.html", history=session["chat_history"], audio_file=audio_file, active="llm")


@app.route("/reset_chat")
def reset_chat():
    session.pop("chat_history", None)
    return redirect(url_for("ask_llm"))


# ========== Run Server ==========
if __name__ == "__main__":
    app.run(debug=True)
