# stt.py
# Real speech-to-text using OpenAI Whisper (local, offline once the model is
# downloaded once). Previously app.py never called this module at all --
# /stt returned a hardcoded string regardless of the uploaded audio.

import os

_model = None


def _get_model():
    """Lazy-load the Whisper model on first use instead of at import time.

    Loading at import time means every route in the Flask app pays Whisper's
    load cost, even ones (like /tts) that never touch speech-to-text. Loading
    it lazily, once, and caching it means only the first STT request is slow.
    """
    global _model
    if _model is None:
        import whisper
        _model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
    return _model


def speech_to_text(audio_path: str) -> str:
    """Transcribe an audio file to text. Returns an error string on failure
    rather than raising, so calling routes can display it directly."""
    try:
        model = _get_model()
        result = model.transcribe(audio_path)
        return result["text"].strip()
    except Exception as e:
        return f"Error transcribing audio: {e}"
