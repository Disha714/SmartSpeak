import pyttsx3
import os
from datetime import datetime

def text_to_speech(text, voice_type='female'):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    # Map the voice type to voice index or name
    if voice_type == 'male':
        for voice in voices:
            if 'male' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
    elif voice_type == 'female':
        for voice in voices:
            if 'female' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
    elif voice_type == 'uk':
        for voice in voices:
            if 'english' in voice.name.lower() and 'uk' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
    elif voice_type == 'indian':
        for voice in voices:
            if 'hindi' in voice.name.lower() or 'indian' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break

    # Save to file
    output_dir = "static/tts"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
    path = os.path.join(output_dir, filename)
    engine.save_to_file(text, path)
    engine.runAndWait()

    return f"/{path.replace(os.sep, '/')}"
