# Command Line Interface (CLI) installation guide

This guide is for advanced users, who are comfortable in the command prompt
of their respective operating system. If that's not you, check out the installation
guides for [MacOS](install_macos.md) or [Windows](install_windows.md).

This guide uses Docker in the command line only, skipping the GUI of Docker Desktop.

1. Install `docker`

Here is the [official installation guide](https://docs.docker.com/engine/install/),
and additionally there will be quite a few tutorials online.


2. Pull the docker image
```
docker pull sinclairhudson/balltracking:latest
```

3. Run the docker image with special options

```
docker run -v VIDEO_IO:/balltracking/io -p 7860:7860 -it --ipc host balltracking
```

Replace `VIDEO_IO` with the folder you'd like to use for input and output.

Optionally, if your docker can access your Nvidia GPU, then this software can make
use of it for running neural networks. This will result in much faster detections,
and a much better user experience.
Instead of the above command, run this very similar one:

```
docker run -v VIDEO_IO:/balltracking/io -p 7860:7860 -it --ipc host --gpus all balltracking
```

4. Go to `localhost:7860` in your web browser
You should see a GUI, like in the image below
![GUI](../img/browser_gradio_interface.PNG)
