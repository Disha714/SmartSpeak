# summarize.py

from transformers import pipeline
from gtts import gTTS
import os

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_with_tts(text):
    if not text or len(text.split()) < 5:
        summary = "The input text is too short for summarization."
    else:
        summary = summarizer(text, max_length=100, min_length=10, do_sample=False)[0]['summary_text']

    # Generate TTS
    tts = gTTS(summary)
    output_path = "output/summary.mp3"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts.save(output_path)

    return {
        "summary": summary,
        "audio_path": output_path
    }
