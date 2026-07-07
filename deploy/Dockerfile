# SlideStream web UI (slide-stream serve) for self-hosting on a VPS.
FROM python:3.12-slim

# FFmpeg is required to encode video; fonts for text slides.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install from PyPI with the web UI + all cloud AI provider clients.
# (Local/offline extras like [local-tts] are large; add them if you want
# offline TTS baked into the image.)
ARG SLIDE_STREAM_VERSION=
RUN pip install --no-cache-dir "slide-stream[all-ai,serve]${SLIDE_STREAM_VERSION:+==$SLIDE_STREAM_VERSION}"

# Non-root user; a home for the layered server config (~/.slidestream.yaml)
# and the model/asset caches.
RUN useradd -m app
USER app
WORKDIR /home/app

EXPOSE 8080

# Config, secrets, token, and the demo flag come from the environment
# (docker-compose .env). Binds 0.0.0.0 so the container is reachable.
CMD ["slide-stream", "serve", "--host", "0.0.0.0", "--port", "8080", "--no-browser"]
