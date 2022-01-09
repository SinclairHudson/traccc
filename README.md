# ball-tracking

Tracking sports balls

# Model training quickstart

This project uses a docker image built off of an NVIDIA image, to allow for 
training models on GPUs.
First, build the docker image (will take some time at the start)
```
docker build -t ball-tracking .
```


Next, spin up the docker container, and point it to the COCO dataset

```
bash training_start.sh /datasets/COCO
```


