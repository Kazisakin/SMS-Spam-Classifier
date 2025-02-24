# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files to the container
COPY . .

# Create a virtual environment and activate it
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --break-system-packages -r requirements.txt

# Expose the application port
EXPOSE 8080

# Start the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app.app:app"]
