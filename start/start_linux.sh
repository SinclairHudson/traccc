#!/bin/bash
set -e

current=`pwd`
docker run -v ~/balltracking_io:/balltracking/io -p 7860:7860 -it --ipc host --gpus all balltracking
#docker run -v $current:/workspace -it --ipc host ball-tracking /bin/bash
