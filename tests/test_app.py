# tests/test_app.py
#
# These tests exercise route wiring, request handling, and session/state
# logic WITHOUT invoking real Whisper/BLIP/sentence-transformer models or
# real network calls (gTTS, Groq, Google Translate). That's a
# deliberate choice: those models are slow to load and need network access
# to download weights on first run, which would make the test suite slow
# and flaky, and isn't what these tests are meant to verify. Model behavior
# itself (does Whisper transcribe correctly, etc.) is out of scope here --
# that's the library's own test surface, not this app's.
#
# Run with: pytest tests/test_app.py

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import app as app_module  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    # Stub out every heavy/network-dependent call before the test client
    # makes a single request.
    monkeypatch.setattr(app_module, "speak", lambda text, lang="en": "fake_audio.mp3")
    monkeypatch.setattr(app_module.stt, "speech_to_text", lambda path: "fake transcript")
    monkeypatch.setattr(
        app_module.translator, "translate_text", lambda text, lang: f"[{lang}] {text}"
    )
    monkeypatch.setattr(app_module.summarize, "summarize_text", lambda text: "fake summary")
    monkeypatch.setattr(
        app_module.sentiment_tts, "analyze_sentiment", lambda text: ("Positive", 0.8)
    )
    monkeypatch.setattr(
        app_module.sentiment_tts,
        "speak_with_emotion",
        lambda text, polarity, output_dir="static": "fake_sentiment.mp3",
    )
    monkeypatch.setattr(app_module.image_caption, "caption_image", lambda path: "a fake caption")
    monkeypatch.setattr(app_module.rag_chat, "build_messages", lambda q, h: [{"role": "user", "content": q}])
    monkeypatch.setattr(app_module.rag_chat, "add_turn", lambda q, a: None)

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_home_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"SmartSpeak" in resp.data


def test_tts_route(client):
    resp = client.post("/tts", data={"text": "hello world", "voice": "us"})
    assert resp.status_code == 200
    assert b"fake_audio.mp3" in resp.data


def test_sentiment_route_shows_polarity(client):
    resp = client.post("/sentiment", data={"text": "I love this!"})
    assert resp.status_code == 200
    assert b"Positive" in resp.data
    assert b"0.8" in resp.data


def test_stt_route_transcribes_upload(client):
    data = {"audio": (io.BytesIO(b"fake audio bytes"), "test.wav")}
    resp = client.post("/stt", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"fake transcript" in resp.data


def test_summarize_route_uses_real_transcript_not_hardcoded_string(client):
    data = {"audio": (io.BytesIO(b"fake audio bytes"), "test.wav")}
    resp = client.post("/summarize", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"fake transcript" in resp.data
    assert b"fake summary" in resp.data
    # The old hardcoded placeholder must be gone.
    assert b"demonstrate audio processing" not in resp.data


def test_translate_route_is_not_reversed_text(client):
    resp = client.post("/translate", data={"text": "hello", "target_lang": "fr"})
    assert resp.status_code == 200
    assert b"[fr] hello" in resp.data
    # The old fake implementation reversed the string -- make sure that's gone.
    assert b"olleh" not in resp.data


def test_image_route(client):
    data = {"image": (io.BytesIO(b"fake image bytes"), "test.jpg")}
    resp = client.post("/image", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"a fake caption" in resp.data


def test_llm_route_without_api_key_shows_clear_error(client, monkeypatch):
    monkeypatch.setattr(app_module, "GROQ_API_KEY", None)
    with client.session_transaction() as sess:
        sess["chat_history"] = []
    resp = client.post("/llm", data={"prompt": "hi"})
    assert resp.status_code == 200
    assert b"GROQ_API_KEY is not set" in resp.data


def test_reset_chat_clears_session(client):
    with client.session_transaction() as sess:
        sess["chat_history"] = [{"question": "hi", "answer": "hello", "timestamp": "12:00:00"}]
    resp = client.get("/reset_chat", follow_redirects=True)
    assert resp.status_code == 200
    with client.session_transaction() as sess:
        assert sess.get("chat_history") == []
