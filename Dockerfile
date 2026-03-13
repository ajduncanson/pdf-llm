FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY governance_config.yaml .
COPY pdf_llm/ ./pdf_llm/

ENTRYPOINT ["python", "main.py"]
