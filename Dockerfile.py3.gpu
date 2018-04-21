FROM tensorflow/tensorflow:latest-gpu-py3

ENV TENSORFLOW_GPU true
WORKDIR /opt/chiron
COPY . .
RUN ["python", "setup.py", "install"]

WORKDIR /data

