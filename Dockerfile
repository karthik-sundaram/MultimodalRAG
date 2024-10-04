

FROM ubuntu:22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    redis-server \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY app/ ./app
COPY data/ ./data

EXPOSE 8501

COPY scripts/start.sh ./start.sh
RUN chmod +x ./start.sh

CMD ["./start.sh"]
