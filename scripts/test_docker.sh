#!/bin/bash

# Exit on error
set -e

echo "Testing MedSynx Docker Setup..."
echo "==============================="

# Build the image
echo "1. Building Docker image..."
docker build -t medsynx:test -f docker/Dockerfile .

# Run tests in container
echo "2. Running tests in container..."
docker run --rm medsynx:test pytest

# Test API endpoints
echo "3. Testing API endpoints..."
docker run -d --name medsynx_test -p 8000:8000 medsynx:test
sleep 5  # Wait for server to start

# Test health endpoint
echo "Testing health endpoint..."
curl -f http://localhost:8000/health

# Test upload endpoint
echo "Testing upload endpoint..."
curl -X POST -F "file=@sample_data/demographics.csv" http://localhost:8000/api/upload

# Test generation endpoint
echo "Testing generation endpoint..."
curl -X POST -F "file=@sample_data/demographics.csv" -F 'config={"epsilon":1.0,"delta":1e-5,"numSamples":100,"noiseMultiplier":1.0}' http://localhost:8000/api/generate

# Clean up
echo "4. Cleaning up..."
docker stop medsynx_test
docker rmi medsynx:test

echo "==============================="
echo "Docker tests completed successfully!" 