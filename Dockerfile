FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app/ ./
COPY frontend/ ./frontend/
COPY data/ ./data/

# Cloud Run expects port 8080
ENV PORT=8080
EXPOSE 8080

# Serve static files from /app/frontend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
