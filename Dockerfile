# syntax=docker/dockerfile:1
# Multi-arch base: builds natively on the Raspberry Pi 5 (linux/arm64).
# Every dependency ships prebuilt aarch64 wheels, so no compiler stage is needed.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg/ffprobe do all decoding, time-stretching and MP3 encoding
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --chown=app:app mixer.py bot.py ./

USER app
STOPSIGNAL SIGTERM
CMD ["python", "-u", "bot.py"]
