FROM python:3.10-slim

# system libs for dlib/face_recognition
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT=8000
EXPOSE 8000
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "--bind", "0.0.0.0:8000", "app:app"]
