# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for compiling some Python packages like pandas)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (optimizes caching)
COPY requirements.txt .

# Upgrade pip and install dependencies directly (no virtualenv needed in Docker)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Set environment variables for Flask and app configuration
ENV FLASK_ENV=production \
    PORT=8080 \
    DATA_DIR=/data \
    PYTHONUNBUFFERED=1 

# Expose the port Gunicorn will run on
EXPOSE 8080

# Start the application using Gunicorn with optimized settings
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--timeout", "0", "--bind", "0.0.0.0:8080", "app.app:app"]