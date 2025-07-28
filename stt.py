# stt.py

import whisper
import os
from datetime import datetime

model = whisper.load_model("base")

def speech_to_text(audio_path):
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error: {str(e)}"
