FROM nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip && \
    python3 -m pip install fastapi uvicorn numpy opencv-python-headless python-multipart

COPY app.py /app/app.py
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
