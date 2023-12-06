#!/bin/bash
set -e

current=`pwd`
# mount just the io when using the app, this is for development.
#docker run -v $current:/balltracking -p 7860:7860 -it --ipc host --gpus all balltracking
docker run -v $current:/balltracking -p 7860:7860 -it --ipc host --gpus all balltracking
#docker run -v ~/balltracking_io:/balltracking/io -p 7860:7860 -it --ipc host --gpus all balltracking
