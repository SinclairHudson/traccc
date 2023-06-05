import skvideo.io
import argparse
from detectors import RawPretrainedDetector, HuggingFaceDETR
from torchvision.io import read_video, VideoReader
import torch



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracks objects using detections as input.")
    parser.add_argument("name", help="name of the project to be tracked.")
    parser.add_argument("--model", help="choice of model", default="DETR")
    args = parser.parse_args()
    name = args.name

    model_selector = {
        "DETR": HuggingFaceDETR,
        "Pretrained": RawPretrainedDetector
    }

    vid_generator = skvideo.io.vreader(f"io/{name}.mp4")
    detector = model_selector[args.model]()
    detector.detect(vid_generator, filename=f"internal/{name}.npz")






