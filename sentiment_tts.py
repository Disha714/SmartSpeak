# sentiment_tts.py

from textblob import TextBlob
from gtts import gTTS
import os

def sentiment_to_speech(text):
    # Analyze sentiment
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    # Choose response based on sentiment
    if polarity > 0:
        response = "You sound positive! " + text
    elif polarity < 0:
        response = "You seem a bit negative. " + text
    else:
        response = "That sounds neutral. " + text

    # Generate speech
    tts = gTTS(response)
    output_path = "output/sentiment_output.mp3"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts.save(output_path)

    return output_path
