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
    args = parser.parse_args()
    name = args.name

    model_selector = {
        "DETR": HuggingFaceDETR,
        "Pretrained": PretrainedRN50Detector
    }

    vid_generator = skvideo.io.vreader(f"io/{name}.mp4")
    metadata = skvideo.io.ffprobe(f"io/{name}.mp4")
    frame_count = int(metadata['video']['@nb_frames'])

    detector = model_selector[args.model]()
    detector.detect(vid_generator, filename=f"internal/{name}.npz", frame_count=frame_count)
