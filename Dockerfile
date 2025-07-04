FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install filterpy transformers timm sk-video opencv-python pylint
RUN pip install pytest
RUN conda install -c conda-forge gradio  # both are needed, for docker build to run
RUN pip3 install gradio==3.40.0
RUN echo 'alias py3="python3"' >> ~/.bashrc
RUN echo 'alias python="python3"' >> ~/.bashrc

RUN mkdir -p /balltracking/io
RUN mkdir -p /balltracking/internal

WORKDIR /balltracking
ADD *.py /balltracking/
ADD traccc/*.py /balltracking/traccc/
CMD ["python3", "gradio_app.py"]
EXPOSE 7860
