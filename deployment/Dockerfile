FROM python:3.10-slim

LABEL maintainer="your.email@university.edu"
LABEL description="AI-Powered Network Threat Hunter"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpcap-dev \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p /app/logs /app/models /app/data

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["python", "demo.py"]