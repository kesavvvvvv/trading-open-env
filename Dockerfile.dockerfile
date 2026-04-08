# Base image
FROM python:3.11-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose port (MANDATORY for HF Spaces)
EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "src.aitea.api.app:app", "--host", "0.0.0.0", "--port", "7860"]