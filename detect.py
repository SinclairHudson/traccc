import skvideo.io
from detectors import RawPretrainedDetector
from torchvision.io import read_video
import torch



if __name__ == "__main__":
    print("reading video")
    vid = skvideo.io.vread("io/bussin.mp4", num_frames=200)
    detector = RawPretrainedDetector(device="cuda")
    detector.detect(torch.Tensor(vid), filename="io/living_room.npz")






