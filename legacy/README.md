# Legacy prototypes

These are earlier iteration scripts (CLI version, a Gradio version, a bare chatbot
script) from before the project was consolidated into the single Flask app in
`backend/app.py`. Kept for history, not maintained, not imported by the app.

`ai_tts_final.py` previously had a live Together.ai API key hardcoded in it — it's
been replaced with an env var read. If this repo was ever pushed to a public
GitHub remote with the key in it, rotate that key on together.ai; scrubbing it
from the latest commit doesn't remove it from git history.

`tts_pyttsx3_unused.py` (originally `tts.py`) was never actually called by
`app.py` — dead code. It's also built on pyttsx3's espeak driver, which is
broken on current espeak/espeak-ng builds (see the comment at the top of
`backend/sentiment_tts.py` for details). The app's real speech synthesis path
is `app.py`'s `speak()` (gTTS), with `sentiment_tts.py` doing gTTS + pydub
post-processing for emotion-driven prosody.
