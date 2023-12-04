#!/bin/bash
set -e

current=`pwd`
docker run -v $current:/balltracking -it -p 7860:7860 --ipc host balltracking
