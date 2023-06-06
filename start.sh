#!/bin/bash
set -e

current=`pwd`
docker run -v $current:/workspace/balltracking -it --ipc host --gpus all ball-tracking /bin/bash
#docker run -v $current:/workspace -it --ipc host ball-tracking /bin/bash
