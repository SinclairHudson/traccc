#!/bin/bash
set -e

current=`pwd`
docker run -v $current:/workspace -it --ipc host --gpus all ball-tracking /bin/bash
