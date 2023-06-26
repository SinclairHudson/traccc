#!/bin/bash
set -e

current=`pwd`
docker run -v $current:/balltracking -it --ipc host --gpus all balltracking /bin/bash
#docker run -v $current:/workspace -it --ipc host ball-tracking /bin/bash
