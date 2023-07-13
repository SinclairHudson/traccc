import argparse

import skvideo.io
import torch
from torchvision.io import VideoReader, read_video

from detectors import HuggingFaceDETR, PretrainedRN50Detector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tracks objects using detections as input.")
    parser.add_argument("name", help="name of the project to be tracked.")
    parser.add_argument("--model", help="choice of model", default="DETR")
    parser.add_argument("--input", help="video file to be used", default=None)
    args = parser.parse_args()
    name = args.name
    input_file = args.input
    if input_file is None:
        input_file = f"io/{name}.mp4"

    model_selector = {
        "DETR": HuggingFaceDETR,
        "RN50": PretrainedRN50Detector
    }

    vid_generator = skvideo.io.vreader(input_file)
    metadata = skvideo.io.ffprobe(input_file)
    frame_count = int(metadata['video']['@nb_frames'])

    detector = model_selector[args.model]()
    detector.detect(vid_generator, filename=f"internal/{name}.npz", frame_count=frame_count)
