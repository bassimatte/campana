FROM python:3.11-slim

WORKDIR /app

# System deps for scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Application
COPY . .

# Render provides PORT env variable
ENV PORT=10000
EXPOSE ${PORT}

CMD uvicorn engine.web_server:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 300
