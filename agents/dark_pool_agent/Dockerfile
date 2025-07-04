FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for multi-tool integration
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY . .

# Create necessary directories for multi-tool integration
RUN mkdir -p /app/data /app/logs /app/cache

# Set environment variables for multi-tool integration
ENV PYTHONPATH=/app
ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
ENV HAYSTACK_DEFAULT_PIPELINE_YAML_PATH=/app/pipeline.yaml
ENV LLAMA_INDEX_CACHE_DIR=/app/cache

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the agent
CMD ["python", "main.py"] 