FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV + Streamlit
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/       src/
COPY configs/   configs/
COPY server.py  server.py
COPY app.py     app.py

ENV PYTHONPATH=/app
ENV API_BASE=http://localhost:8000

# HuggingFace Spaces runs as non-root user 1000
RUN useradd -m -u 1000 hfuser && chown -R hfuser:hfuser /app
USER 1000

EXPOSE 7860

# Start FastAPI backend in background, then launch Streamlit on port 7860
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 8000 &\
     sleep 4 &&\
     streamlit run app.py --server.port 7860 --server.address 0.0.0.0\
       --server.headless true --browser.gatherUsageStats false"]
