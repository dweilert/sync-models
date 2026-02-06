FROM python:3.12-slim

# System deps for rsync + ssh
RUN apt-get update && apt-get install -y --no-install-recommends \
    rsync \
    openssh-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} -s /bin/bash app

WORKDIR /app
COPY sync-models.py /app/sync-models.py

# Python deps (pin if you want)
RUN pip install --no-cache-dir fastapi uvicorn requests pydantic

USER app
ENV HOME=/home/app

EXPOSE 9090
CMD ["python", "/app/sync-models.py", "--host", "0.0.0.0", "--port", "9090", "--peer-poll-seconds", "10"]
