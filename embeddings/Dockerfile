FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip install --no-cache-dir \
    torch \
    transformers \
    fastapi \
    uvicorn \
    numpy

# Copy the embeddings service script
COPY embeddings_service.py /app/embeddings_service.py

# Set the working directory
WORKDIR /app

# Expose the port for the embeddings service
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "embeddings_service:app", "--host", "0.0.0.0", "--port", "8000"]
