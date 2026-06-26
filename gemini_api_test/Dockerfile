FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# GEMINI_API_KEY supplied at run time (env). Set GEMINI_MOCK=1 to run without a key.
CMD ["python", "app.py"]
