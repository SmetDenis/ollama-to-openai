# Use Python 3.13 slim image as base
FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY . /app
RUN uv sync --frozen

EXPOSE 11434
CMD [".venv/bin/python", "ollama_to_openai_adapter.py"]
