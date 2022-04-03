import skvideo.io
from detectors import RawPretrainedDetector
from torchvision.io import read_video
import torch



if __name__ == "__main__":
    print("reading video")
    vid = skvideo.io.vread("io/living_room.mp4")
    detector = RawPretrainedDetector(device="cuda")
    print("showing detections")
    detector.detect(torch.Tensor(vid), filename="io/living_room.npz")






