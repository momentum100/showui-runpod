# Use a base image with PyTorch and CUDA
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to start your RunPod handler
CMD ["python", "handler.py"]
