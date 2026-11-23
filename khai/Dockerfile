FROM python:3.12-slim
COPY . /src
RUN pip install --no-cache-dir "/src[apps]" && rm -rf /src
