# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /code
WORKDIR /code

# Install system dependencies required for OpenCV
# libgl1 is the modern replacement for libgl1-mesa-glx in newer Debian versions
# libglib2.0-0 is required for some GLib functionality used by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code
COPY . /code

# Define the command to run the app
# Hugging Face Spaces expects the app to run on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
