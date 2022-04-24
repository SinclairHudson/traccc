FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install filterpy transformers timm sk-video opencv-python
