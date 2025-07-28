from gtts import gTTS
import os

# Your text input
text = "Hello! This is a text to speech demo using gTTS."

# Language (you can change 'en' to 'hi' for Hindi, etc.)
tts = gTTS(text=text, lang='en')

# Save the audio to a file
tts.save("output.mp3")

# Play the audio file (Windows only)
os.system("start output.mp3")
