# Dockerfile.ml — Python ML inference services (YOLO, SegFormer)
FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY python/ python/
COPY config/ config/
COPY environment.yml .

RUN pip3 install --no-cache-dir \
    ultralytics \
    transformers \
    torch \
    torchvision \
    onnxruntime \
    numpy \
    opencv-python-headless

ENTRYPOINT ["/bin/bash", "-c", ". /opt/ros/humble/setup.sh && exec \"$@\"", "--"]
