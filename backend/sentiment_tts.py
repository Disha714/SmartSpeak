# sentiment_tts.py
# Emotion-aware speech: previously this just spoke "The sentiment is X" as a
# flat label -- it never actually changed how the speech sounded.
#
# This version generates real speech via gTTS, then post-processes it with
# pydub so a positive line is actually spoken faster and louder, and a
# negative line slower and quieter, instead of just narrating the sentiment
# as a label.
#
# Note on an approach we deliberately did NOT take: pyttsx3 (local,
# offline TTS) would let us set rate/volume before synthesis instead of
# after. It was tried first, but pyttsx3's espeak driver is unmaintained and
# breaks on current espeak/espeak-ng builds (they now use hierarchical voice
# IDs like "gmw/en" that the driver's hardcoded default voice lookup can't
# resolve) -- a known, currently-unresolved compatibility issue, not
# something fixable from this codebase. Post-processing the gTTS output with
# pydub sidesteps it entirely and is testable without a local TTS engine.

import os
import time

from gtts import gTTS
from pydub import AudioSegment
from textblob import TextBlob


def _tone_for(polarity: float) -> str:
    if polarity > 0.4:
        return "Positive"
    if polarity < -0.4:
        return "Negative"
    return "Neutral"


def analyze_sentiment(text: str):
    """Returns (tone, polarity) for the given text."""
    polarity = TextBlob(text).sentiment.polarity
    return _tone_for(polarity), polarity


def _apply_prosody(audio: AudioSegment, polarity: float) -> AudioSegment:
    """Shift playback speed and loudness based on polarity in [-1, 1].
    Speed: 0.75x (very negative) to 1.25x (very positive).
    Gain: -6dB (very negative) to +6dB (very positive).
    """
    clamped = max(-1.0, min(1.0, polarity))
    speed_factor = 1.0 + clamped * 0.25
    gain_db = clamped * 6

    # Classic pydub speed-change trick: reinterpret the same raw samples at a
    # different frame rate, then resample back to the original rate so
    # playback duration (and pitch, as a side effect) shifts accordingly.
    shifted = audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * speed_factor)},
    ).set_frame_rate(audio.frame_rate)

    return shifted.apply_gain(gain_db)


def speak_with_emotion(text: str, polarity: float, output_dir: str = "static") -> str:
    """Synthesize `text` with prosody scaled by `polarity`.
    Returns the filename (relative to output_dir) of the generated audio."""
    os.makedirs(output_dir, exist_ok=True)

    tmp_path = os.path.join(output_dir, f"_sentiment_raw_{int(time.time())}.mp3")
    gTTS(text=text).save(tmp_path)

    audio = AudioSegment.from_mp3(tmp_path)
    processed = _apply_prosody(audio, polarity)

    filename = f"sentiment_{int(time.time())}.mp3"
    path = os.path.join(output_dir, filename)
    processed.export(path, format="mp3")

    os.remove(tmp_path)
    return filename
