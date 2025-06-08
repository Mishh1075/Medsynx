FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Copy and set permissions for the startup script
COPY run.py .
RUN chmod +x run.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "run.py"] 