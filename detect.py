import cv2
from detectors import RawPretrainedDetector
from torchvision.io import read_video



if __name__ == "__main__":
    print("reading video")
    vid, audio, fps = read_video("io/test.mp4")  # THWC
    detector = RawPretrainedDetector(device="cuda")
    print("showing detections")
    detector.detect(vid)






