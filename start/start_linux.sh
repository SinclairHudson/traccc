#!/bin/bash
set -e

current=`pwd`
# mount just the io when using the app, this is for development.
docker run -v $current:/balltracking -p 7860:7860 -it --ipc host --gpus all balltracking
