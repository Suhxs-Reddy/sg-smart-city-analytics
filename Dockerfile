FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    streamlit==1.32.0 \
    streamlit-folium==0.18.0 \
    streamlit-autorefresh==1.0.1 \
    folium==0.16.0 \
    requests==2.31.0 \
    Pillow==10.0.0

COPY app.py .

# HuggingFace Spaces runs as non-root uid 1000
RUN useradd -m -u 1000 hfuser
USER 1000

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
