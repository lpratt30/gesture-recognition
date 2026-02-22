# 1. Use NVIDIA CUDA base (matches your ML focus)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 2. Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install System Dependencies 
# Note: OpenCV and PyGame need these extra libraries to handle images and windows
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Workspace
WORKDIR /app

# 5. Copy and Install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. Copy code
COPY . .

# 7. Start the pipeline
CMD ["python3", "main.py"]