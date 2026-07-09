# SmartSpeak: AI-Powered Speech Generation System

SmartSpeak converts text, sentiment, images, and speech into natural speech
output, combining NLP, computer vision, and retrieval-augmented generation
into one Flask app.

---

## Core Features

1. **Text-to-Speech** — text in, spoken audio out, with voice/locale options.
2. **Speech-to-Text** — real transcription via OpenAI Whisper (previously this
   route returned a hardcoded placeholder string regardless of the uploaded
   audio; now it actually transcribes it).
3. **Sentiment-to-Speech** — detects sentiment (TextBlob), then actually
   modulates the speech's speed and loudness based on the polarity score,
   instead of just reading the sentiment label aloud.
4. **Speech Summarization** — transcribes uploaded audio (Whisper), then
   summarizes the real transcript (distilbart-cnn-12-6) — previously this
   summarized a hardcoded demo sentence regardless of the uploaded file.
5. **Translate & Speak** — real translation (deep-translator) — previously
   this reversed the input string and called it "translated."
6. **Image-to-Speech** *(new)* — upload an image, get a spoken caption (BLIP
   captioning). Turns the project from a text/audio tool into a multimodal
   one.
7. **RAG-augmented Ask AI Chat** — retrieves the most relevant past turns
   from a FAISS vector index (via sentence-transformers embeddings) instead
   of stuffing the entire raw conversation history into every prompt. Chat
   completions run on Groq (free tier, no credit card required, OpenAI-
   compatible API).

---

## Tech Stack

- **Backend**: Python, Flask, Flask-Session
- **Speech**: gTTS (synthesis), OpenAI Whisper (transcription), pydub (prosody
  post-processing for the sentiment feature)
- **NLP/AI**: TextBlob (sentiment), Transformers (summarization, BLIP
  captioning), Groq (Llama 3.3 70B chat completions, free tier)
- **RAG**: sentence-transformers + FAISS
- **Translation**: deep-translator
- **Frontend**: HTML, CSS (Jinja2 templates)

---

## Project Structure

```
SmartSpeak/
├── backend/
│   ├── app.py              # Flask app, all routes
│   ├── stt.py               # Whisper transcription
│   ├── translator.py        # deep-translator wrapper
│   ├── summarize.py         # transformers summarization
│   ├── sentiment_tts.py      # sentiment analysis + emotion-modulated speech
│   ├── rag_chat.py          # FAISS + sentence-transformers retrieval
│   ├── image_caption.py      # BLIP image captioning
│   ├── requirements.txt
│   └── .env.example
├── templates/                # Jinja2 templates for every route
├── static/                   # CSS + generated audio/uploads (gitignored)
├── tests/
│   └── test_app.py           # route-level tests, heavy models mocked out
├── legacy/                   # earlier prototype scripts, not used by the app
├── Dockerfile
└── README.md
```

---

## How to Run Locally

```bash
git clone https://github.com/Disha714/SmartSpeak.git
cd SmartSpeak

cp backend/.env.example backend/.env
# then edit backend/.env and add a real GROQ_API_KEY (free at console.groq.com)

pip install -r backend/requirements.txt
python backend/app.py
```

Open http://127.0.0.1:5000/ in your browser.

The first request to `/stt`, `/summarize`, or `/image` will download the
relevant model (Whisper / distilbart / BLIP) from Hugging Face — this needs
an internet connection the first time, then the model is cached locally.

## Run with Docker

```bash
docker build -t smartspeak .
docker run -p 5000:5000 --env-file backend/.env smartspeak
```

## Run the tests

```bash
pip install pytest
pytest tests/test_app.py -v
```

These tests mock out Whisper/BLIP/sentence-transformers/gTTS/Groq so
they run in seconds without downloading models or needing network access —
they check routing, request handling, and session logic, not model output
quality.
