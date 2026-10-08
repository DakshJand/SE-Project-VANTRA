FROM python:3.12-slim

WORKDIR /app

# system deps: libpq for psqlpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev fonts-dejavu-core ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY docker-entrypoint.sh .

# NOTE: the sample traffic clip for video-live mode lives in data/ (gitignored),
# so it is not copied here. Regenerate with `python scripts/make_traffic_video.py`
# and bind-mount ./data/videos into the container to enable that mode.

# default to the compose service name for the DB
ENV VANTRA_DATABASE_URL=postgresql://vantra:vantra@db:5432/vantra

EXPOSE 8712

# wait for DB, init schema, seed on first boot (idempotent), then serve
ENTRYPOINT ["./docker-entrypoint.sh"]
