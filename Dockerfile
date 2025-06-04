# Use Miniconda as the base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Create Conda environment
RUN conda create -n lip-sync-api python=3.9

# Activate Conda environment for subsequent commands
SHELL ["conda", "run", "-n", "lip-sync-api", "/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY main.py .
COPY Wav2Lip/ ./Wav2Lip/
COPY models/ ./models/

# Expose port 8000 for the WebSocket server
EXPOSE 8000

# Run the FastAPI server
CMD ["conda", "run", "-n", "lip-sync-api", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]