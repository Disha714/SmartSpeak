# summarize.py
# Single, consistent summarization model used everywhere (previously app.py
# used t5-small while this file used distilbart-cnn-12-6 -- two different
# models for the "same" feature depending on which code path ran).

_summarizer = None


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return _summarizer


def summarize_text(text: str) -> str:
    """Summarize arbitrary text. Guards against inputs too short to
    meaningfully summarize (the underlying model errors on very short input)."""
    if not text or len(text.split()) < 8:
        return text.strip() if text else "No text to summarize."
    summarizer = _get_summarizer()
    result = summarizer(text, max_length=100, min_length=10, do_sample=False)
    return result[0]["summary_text"]
