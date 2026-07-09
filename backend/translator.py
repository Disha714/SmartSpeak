# translator.py
# Real translation using deep-translator (wraps the Google Translate web
# endpoint without the API-breakage issues `googletrans` has had). The old
# version of this file did `text[::-1]` and called it "translated" -- that's
# gone.

from deep_translator import GoogleTranslator


def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target_lang (e.g. 'fr', 'hi', 'es').
    Returns an error-prefixed string on failure instead of raising, so the
    Flask route can render it directly without a try/except at the call site.
    """
    if not text or not text.strip():
        return ""
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        return f"Error translating text: {e}"
