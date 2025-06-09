# Use multi-stage build for smaller final image
FROM python:3.8-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model weights
RUN mkdir -p /app/weights && \
    curl -L https://github.com/yourusername/medsynx/releases/download/v1.0/model_weights.tar.gz | \
    tar -xz -C /app/weights

# Final stage
FROM python:3.8-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy model weights
COPY --from=builder /app/weights /app/weights

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 medsynx
USER medsynx

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ app/
COPY scripts/ scripts/
COPY alembic/ alembic/
COPY alembic.ini .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 