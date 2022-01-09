# Project Goals

This project aims to create an application that tracks sports balls (tennis, basketball, volleyball, etc) through the air,
using some object detector trained on the COCO dataset, and kalman filters to track. Additionally, this application will
be able to add different effects to each track, to visualize them in a video format.

## Structure

There are 4 parts to the project:
* Model training on the COCO dataset
* Model inference on individual frames of an input image
* Tracking objects based on detections
* adding effects based on tracks

The 4 parts should be as de-coupled as possible. Specifically, we want to save
computation and only do the object detections once for a single image.

## Technology and frameworks

The whole project will be dockerized with cuda support (and cpu support), to make
the application easy to run for users as well as developers.

For anything machine learning, PyTorch
For most arrays, numpy
For kalman filters, pykalman
For effects, image and video I/O, OpenCV
For communication between different parts of the application, YAML files.


## Specifications

### Model training
The model training component just needs to produce some sort of object detection
model, trained exclusively to detect the "sports ball" class in the coco dataset.
It won't be part of the app itself, but should operate in the same docker container.

### Model Inference
Input: User video, in some standard IO directory
Output: YAML file containing all the detections of the entire video. 
Probably some dictionary where every key is a frame and the entries are lists 
of coordinates in the image space.
Example of running inference should be something like

```
python3 detect.py --model model.pth --video input.mp4 --name unique
```

### Object tracking
Input: yaml file generated from the model inference section (detections)
Output: YAML of tracks associated with the detections
Every track has:
* Unique ID
* starting frame (frame at which it appears first)
* list of positions in the image space

```
python3 track.py --detections unique_detections.yaml
```

### Effects
Input: A video file and the tracks associated with them. The video file name should
be in the yaml file already though.
Also need some kind of user input to specify which effects to apply. 
An effect is a function that takes in a track and a reference to the video file and
modifies the video file, applying an affect on the track.
Output: video file with all the effects on each of the tracks, in IO directory

```
python3 add_effects.py --effect sparkles --tracks unique_tracks.yaml
```


