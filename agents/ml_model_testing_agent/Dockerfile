FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for multi-tool integration and ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Expose port
EXPOSE 8000

# Enhanced health check with multi-tool validation
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; import json; \
    response = requests.get('http://localhost:8000/health', timeout=5); \
    data = response.json(); \
    assert data['status'] == 'healthy'; \
    assert data['health_score'] > 0.5; \
    print('Health check passed')" || exit 1

# Run the agent with multi-tool integration
CMD ["python", "main.py"] 