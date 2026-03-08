FROM python:3.11-slim

# Set up a non-root user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Switch to root temporarily to install system dependencies required by OpenCV
USER root
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Fix permissions on /app for the non-root user
RUN chown -R user:user /app
USER user

# Copy the requirements file and install dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=user:user . .

# Ensure the upload and output directories exist
RUN mkdir -p uploads outputs

# Hugging Face Spaces expose port 7860
EXPOSE 7860

# Start the Flask backend
CMD ["python", "app.py"]
