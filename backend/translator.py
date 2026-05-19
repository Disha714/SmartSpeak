from googletrans import Translator
from gtts import gTTS
import tempfile
import os
import pygame

def translate_and_speak(text, target_lang):
    translator = Translator()
    translated = translator.translate(text, dest=target_lang)
    translated_text = translated.text

    tts = gTTS(text=translated_text, lang=target_lang)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_path = fp.name
        tts.save(temp_path)

    # Play audio using pygame
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play()

    # Wait until playback finishes
    while pygame.mixer.music.get_busy():
        continue

    pygame.quit()
    os.remove(temp_path)

    return translated_text
