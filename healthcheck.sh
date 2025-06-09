#!/bin/sh

# Function to check HTTP endpoint
check_http() {
    local url=$1
    local expected_status=$2
    local timeout=$3

    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout" "$url")
    
    if [ "$response" = "$expected_status" ]; then
        return 0
    else
        return 1
    fi
}

# Check service type based on environment variable
case "$SERVICE_TYPE" in
    "api")
        # Check API health endpoint
        check_http "http://localhost:8000/health" "200" "5"
        ;;
    "frontend")
        # Check frontend static files
        check_http "http://localhost:80" "200" "5"
        ;;
    "worker")
        # Check if Celery worker is responding
        celery -A app.worker inspect ping
        ;;
    *)
        echo "Unknown service type: $SERVICE_TYPE"
        exit 1
        ;;
esac 