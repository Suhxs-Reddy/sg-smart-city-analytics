FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit \
    streamlit-folium \
    streamlit-autorefresh \
    folium \
    requests \
    Pillow

COPY app.py .

RUN useradd -m -u 1000 hfuser
USER 1000

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
