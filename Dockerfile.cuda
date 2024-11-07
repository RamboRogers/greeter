# Use Python 3.11.10 on Debian
FROM python:3.11.10-slim

# Set working directory
WORKDIR /app

ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0
# Force TensorFlow to see the GPU
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# Add NVIDIA repository and GPG key
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb

# Install CUDA and cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-11-8 \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set timezone
ENV TZ=America/New_York
RUN apt-get update && apt-get install -y \
    tzdata \
    build-essential \
    libffi-dev \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libwebp-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["python", "app.py"]