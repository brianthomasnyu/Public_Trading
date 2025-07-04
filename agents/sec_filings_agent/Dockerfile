FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for LangChain
ENV LANGCHAIN_ENABLED=true
ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_PROJECT=financial_data_aggregation

# Run the LangChain-enhanced agent
CMD ["python", "main.py"] 