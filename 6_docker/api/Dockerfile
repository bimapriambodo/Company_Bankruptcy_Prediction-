FROM python:3.10-slim-buster
WORKDIR /app
COPY requirements.txt .
COPY . .
RUN \
apt-get update && \
apt-get upgrade -y && \
apt-get autoremove -y && \
apt-get clean -y && \
pip install --upgrade pip && \
pip install wheel && \
pip install -r requirements.txt
EXPOSE 8080
CMD uvicorn src.api:app --host 0.0.0.0 --port 8080 --reload