FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY causality_graph/ causality_graph/
RUN pip install --no-cache-dir -e .
