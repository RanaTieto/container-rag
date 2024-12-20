FROM python:3.9-slim

# Install system dependencies required by psycopg2
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn requests psycopg2-binary pydantic numpy

# Copy the backend code
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt

# Set the working directory
WORKDIR /app

# Install additional Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI app's port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
