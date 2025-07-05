# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/VectorSpaceLab/OmniGen2.git /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124

# Install Gradio for web interface
RUN pip install --no-cache-dir gradio

# Create directories for models and outputs
RUN mkdir -p /app/models /app/outputs

# Expose the port the app runs on
EXPOSE 7860

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV HF_HOME=/app/models

# Run the application
CMD ["python", "app_chat.py", "--share"]
