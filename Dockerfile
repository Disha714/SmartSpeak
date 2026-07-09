FROM python:3.11-slim

# ffmpeg: needed by pydub (audio post-processing) and whisper (audio decoding).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY static/ ./static/
COPY templates/ ./templates/

WORKDIR /app/backend
EXPOSE 5000

CMD ["python", "app.py"]
