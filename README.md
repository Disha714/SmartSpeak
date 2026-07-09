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
   using hybrid retrieval (dense FAISS embedding search + BM25 keyword
   search, fused with Reciprocal Rank Fusion) instead of pure embedding
   similarity or stuffing the whole raw history into every prompt. The
   index persists to disk (`backend/rag_store/`), so retrieval survives a
   server restart. Chat completions run on Groq (free tier, no credit card
   required, OpenAI-compatible API).

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
├── evaluation/
│   └── eval_retrieval.py     # dense vs hybrid retrieval quality comparison
├── tests/
│   └── test_app.py           # route-level tests, heavy models mocked out
├── legacy/                   # earlier prototype scripts, not used by the app
├── Dockerfile
└── README.md
```

---

## Evaluating retrieval quality

`evaluation/eval_retrieval.py` measures whether retrieval actually surfaces
the right past turn for a given query, comparing dense-only (FAISS) against
the hybrid (dense + BM25 via Reciprocal Rank Fusion) retrieval the app
actually uses. The test set is split into "semantic" queries (paraphrases
with no shared vocabulary) and "exact_keyword" queries (hinge on a specific
name/ID/number) to show where each retrieval strategy wins or loses, rather
than just reporting a single aggregate number.

```bash
python evaluation/eval_retrieval.py
```

This needs the sentence-transformers model downloaded (internet on first
run only, then cached) and runs against a throwaway index — it never
touches the app's real conversation history in `backend/rag_store/`.

### Measured results

Hit-rate@1 on a 22-turn corpus with deliberate near-duplicate distractors
(four near-identical "ticket #XXXXX" entries, four near-identical invoice
entries, four near-identical pet-visit entries, differing only by an
ID/name):

| Category        | Dense only | Hybrid (weighted RRF) |
|------------------|:----------:|:----------------------:|
| semantic         | 80%        | 80%                     |
| exact_keyword    | 83%        | **100%**                |
| **overall**      | 82%        | **91%**                 |

The one case worth calling out specifically: a query asking about
`ticket #48217` against three near-identical sibling tickets (`#48213`,
`#48221`, `#48226`). Dense embeddings ranked the *wrong* ticket first —
a known weak point, since embedding models have little reason to weight
an arbitrary ID heavily. Plain (unweighted) RRF didn't fix it either: the
correct document sat at dense-rank 2 / sparse-rank 0, while the wrong one
sat at dense-rank 0 / sparse-rank 2 — an exact mirror image, which produces
a genuine mathematical tie under unweighted RRF regardless of corpus size.
The tie was then broken by incidental sort order rather than any real
signal, and it happened to always favor dense.

The fix was a small, evidence-based one: weight BM25 slightly higher than
dense in the fusion (`SPARSE_WEIGHT = 1.3` vs `DENSE_WEIGHT = 1.0` in
`rag_chat.py`), justified directly by the empirical finding above — dense
is measurably less reliable on arbitrary IDs, BM25 is measurably reliable
there, so ties between them shouldn't be split 50/50. That single change
took `exact_keyword` from 83% to 100% with no regression on `semantic`.

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
