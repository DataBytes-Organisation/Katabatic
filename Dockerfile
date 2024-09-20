# GANBLR Dockerfile
FROM python:3.9-slim

# Set working directory to /app
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy model code
COPY Katabatic/Katabatic/katabatic/models/ganblr /app/ganblr

# Expose port 8000 for the model
EXPOSE 8000

# Run command to start the model
CMD ["python", "katabatic/models/ganblr/ganblr_adapter.py"]