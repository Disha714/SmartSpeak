# SmartSpeak: AI-Powered Speech Generation System

SmartSpeak is an AI-based system that converts text, sentiment, and visual inputs into natural speech.  
It integrates Natural Language Processing and Computer Vision to build an end-to-end speech generation pipeline.

---

## Overview

The system is designed to process multiple forms of input:

- Text input for speech synthesis  
- Sentiment-aware tone modulation  
- Image-based input for speech generation  
- NLP utilities such as summarization and translation  

The goal is to provide a unified interface for intelligent speech generation.

---

## Core Functionalities

1. Text-to-Speech  
   - Converts textual input into speech output  
   - Supports configurable voice output  

2. Sentiment Analysis  
   - Detects sentiment in input text  
   - Adjusts speech tone accordingly  

3. Vision-to-Speech  
   - Processes image input using computer vision  
   - Extracts relevant information and converts it into speech  

4. NLP Processing  
   - Text summarization  
   - Language translation  

---

## Tech Stack

- Backend: Python, Flask / FastAPI  
- AI/ML: PyTorch, Hugging Face Transformers  
- Computer Vision: OpenCV  
- Speech: pyttsx3  
- Frontend: HTML, CSS  

---

## Project Structure
SmartSpeak/
│
├── backend/
│ ├── app.py
│ ├── tts.py
│ ├── sentiment_tts.py
│ ├── translator.py
│ ├── summarize.py
│ └── stt.py
│
├── static/
├── templates/
│
├── requirements.txt
└── README.md


---

## How to Run Locally

Follow these steps to run SmartSpeak on your system:

```bash
git clone https://github.com/Disha714/SmartSpeak.git
cd SmartSpeak

pip install -r requirements.txt
python backend/app.py

Once the server is running, open your browser and navigate to:

http://127.0.0.1:5000/
