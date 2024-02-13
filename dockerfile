# app/Dockerfile

FROM python:3.9-slim

WORKDIR /emia__demo

RUN apt-get clean

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone -b demo https://github.com/Interactive-Coventry/EMIA__demo.git .

RUN apt-get update && apt-get install $(cat packages.txt) --fix-missing -y

RUN pip3 install -r requirements.txt

EXPOSE 8531

HEALTHCHECK CMD curl --fail http://localhost:8531/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8531", "--server.address=0.0.0.0"]