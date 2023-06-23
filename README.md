# ball-tracking

A command line interface tool to track sports balls, and add interesting visual effects. 

Running this project is fairly demanding.
A GPU is very helpful for running the neural networks for ball detection quickly.
At minimum: **5GB** disk space.

# Application quickstart

This application uses [docker](www.docker.com).
Docker should be installed, and following that the command:
```
docker build -t balltracking .
```
From the top-level directory.

From there, the docker container needs to be spun up from the image, which is
accomplished by the script:

```
bash start_linux.sh
```

A Windows version of this script is a work in progress.

## The pipeline

before starting the pipeline, determine a _name_ for your project. For the following
examples, I'll be using `yellowball`

### 1. **Detect** all the sports balls in the video, frame by frame

```
py3 detect.py yellowball 
```

### 2. **Track** the sports balls through sequential frames, using the detections

```
py3 track.py yellowball 
```

###  3. **Visualize** the motion of the balls, using the tracks

```
py3 draw.py yellowball --effect red_dot
```

All of the above commands will have multiple options. You can see what the options
do using the `-h` (help) option, such as:
```
py3 detect.py -h
```

---

## Contributing

Contributions are welcome! Open an issue or a pull request, and I'll get to it when I can.
Adding new effects is an easy contribution to make.
