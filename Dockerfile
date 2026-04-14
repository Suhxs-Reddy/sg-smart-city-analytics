FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV / image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# CPU-only torch first (avoids pulling CUDA, keeps image ~800 MB)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Frontend + CATI inference deps
RUN pip install --no-cache-dir \
    streamlit>=1.32.0 \
    streamlit-folium>=0.18.0 \
    streamlit-autorefresh>=1.0.1 \
    folium>=0.16.0 \
    requests>=2.31.0 \
    Pillow>=10.0.0 \
    ultralytics>=8.3.0 \
    huggingface_hub>=0.22.0 \
    numpy>=1.24.0 \
    pandas>=2.1.0

COPY src/ ./src/
COPY app.py .

RUN useradd -m -u 1000 hfuser
USER 1000

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
