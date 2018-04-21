FROM tensorflow/tensorflow:latest-py3

WORKDIR /opt/chiron
COPY . .
RUN ["python", "setup.py", "install"]

WORKDIR /data

