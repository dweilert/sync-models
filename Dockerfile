FROM python:3.12-slim

# System tools: rsync + ssh client (recommended transfer path)
RUN apt-get update && apt-get install -y --no-install-recommends \
    rsync openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY sync-models.py /app/sync-models.py

# Default port (can be overridden by docker-compose)
EXPOSE 9090

# Run the server
CMD ["python", "sync-models.py", "--host", "0.0.0.0", "--port", "9090", "--peer-poll-seconds", "10"]
