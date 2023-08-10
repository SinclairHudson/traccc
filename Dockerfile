FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install filterpy transformers timm sk-video opencv-python pylint
RUN pip install pytest
RUN conda install -c conda-forge gradio
RUN echo 'alias py3="python3"' >> ~/.bashrc
RUN echo 'alias python="python3"' >> ~/.bashrc

RUN mkdir /balltracking


WORKDIR /balltracking
