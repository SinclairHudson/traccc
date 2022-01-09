#!/bin/bash

dir=$(dirname $(realpath -s $0))

docker run -v "$dir:/workspace" -v "$dataset_root:/datasets" -it --gpus all ball-tracking /bin/bash
