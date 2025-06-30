# Traccc User Guide

This is the user guide for the ball tracking software, usually used via the 
GUI in the browser. The normal workflow is to go through the 3 stages (Detect, Track, Draw) sequentially
with a single video. The __Project Name__ must remain the same for all three stages.
The intermediate data is saved between each step. This means that if you don't like the effect you chose,
you can go back and change the effect and re-draw the video. Similarly, if you feel the need to adjust
the tracking parameters, you can go back to that step, re-track all the balls, and then re-draw your effect
with the new output.

## Detection

Currently, there are two supported detectors:
|Name|Speed|Accuracy|Notes|
|---|---|---|
|DETR|1/5|4/5|struggles with small objects|
|RN50|3/5|2/5|quite a few false positives|
|OWLVIT|3/5|2/5|requires a prompt, struggles with small objects|

Since detection only needs to be run once per video, I would recommend DETR for its
higher accuracy, even though it'll take a few more minutes compared to RN50.

## Tracking

There are two options for ball trackers, **Constant Velocity** and **Constant Acceleration**.
The Constant Velocity tracker generally outputs more smooth trajectories, but it's overall less accurate.
The Constant Acceleration tracker tracks objects closer, but at times extrapolates too far and makes sharp corrections.

Here's an example that illustrates the difference between the two:

|Constant Velocity|Constant Acceleration|
|---|---|
|![constant_vel](../img/line_constant_vel.png)|![constant_acc](../img/line_constant_acc.png)|

Using constant velocity is recommended, though if the system is _losing track of balls_
then you might want to try constant acceleration.

## Effects

Here are all the effects listed. Note that not all effects take all arguments.
For example, The effect `fully_connected` doesn't change when the "length" attribute
is changed, because it doesn't have a fixed length.

|Effect Name|Example|Speed|
|---|---|---|
|dot|![dot](../img/dot.png)|5/5|
|lagging_dot|![lagging_dot](../img/lagging_dot.png)|5/5|
|line|![line](../img/line.png)|5/5|
|highlight_line|![highlight_line](../img/highlight_line.png)|2/5|
|neon_line|![neon_line](../img/neon_line.png)|2/5|
|contrail|![contrail](../img/contrail.png)|1/5|
|fully_connected|![fully_connected](../img/fully_connected.png)|4/5|
|fully_connected_neon|![fully_connected_neon](../img/fully_connected_neon.png)|4/5|
|debug|![debug](../img/debug.png)|2/5|

## Best Practices when shooting videos:

1. Use a **high framerate**. Most cameras, even phone cameras, can shoot 60 frames per second now, go as high as you can.
Every frame is additional information for the tracker, so it's better able to track the balls.
2. Use a **fast shutter speed**. This is usually implied by high framerate, but technically different.
A fast shutter speed will **reduce motion blur**, making the balls in every frame look more like balls 
and thus making them easier to detect.
3. Because a fast shutter speed is required, you need a **lot of light**. Filming outdoors is best.
3. Expose for the things you want to track. If the exposure of the camera is too high or too low, the objects will lose detail
and they'll be almost impossible to detect, because they won't look like much of anything. Even lighting is best,
so the camera doesn't have to expose for the highlights or the shadows.
4. The larger the objects are in the video, the better. Zoom in as much as possible.
5. Use a **small file to test ideas**. Some of the effects and detections are very slow to run,
especially without a GPU. It's best to test and tweak an effect on a short clip, iterate fast, and then
apply it to a longer clip once you're sure you have what you want.
