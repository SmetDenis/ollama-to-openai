# Use Python 3.13 slim image as base
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN uv sync --frozen

EXPOSE 11434
CMD [".venv/bin/python", "-m", "ollama_adapter"]
