import skvideo.io
import argparse
from detectors import RawPretrainedDetector, HuggingFaceDETR
from torchvision.io import read_video, VideoReader
import torch



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracks objects using detections as input.")
    parser.add_argument("name", help="name of the project to be tracked.")
    args = parser.parse_args()
    name = args.name

    vid_generator = skvideo.io.vreader(f"io/{name}.mp4")
    detector = HuggingFaceDETR(device="cuda")
    detector.detect(vid_generator, filename=f"internal/{name}.npz")






