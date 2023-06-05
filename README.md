# ball-tracking

Tracking sports balls

Running this project is fairly demanding.
A GPU is very helpful for running the neural networks for ball detection quickly.
At minimum: **5GB** disk space.

# Application quickstart

This application uses [docker](www.docker.com).

## The pipeline

before starting the pipeline, determine a _name_ for your project. For the following
examples, I'll be using `yellowball`

### **Detect** all the sports balls in the video, frame by frame

```
py3 detect.py yellowball 
```

### **Track** the sports balls through sequential frames, using the detections

```
py3 track.py yellowball 
```

###  **Visualize** the motion of the balls, using the tracks

```
py3 draw.py yellowball --effect red_dot
```

---

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


