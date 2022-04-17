import argparse
import numpy as np
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
import skvideo.io
import torch
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracks objects using detections as input.")
    parser.add_argument("name", help="name of the project to be tracked.")
    args = parser.parse_args()
    name = args.name
    print("reading video")
    vid_generator = skvideo.io.vreader(f"io/{name}.mp4")
    vid_writer = skvideo.io.FFmpegWriter(f"io/{name}_detections.mp4")

    detections = np.load(f"internal/{name}.npz")
    detections_list = []
    for frame_number, frame_name in tqdm(enumerate(detections)):
        detections_list.append(detections[frame_name])

    for i, frame in enumerate(vid_generator):
        if len(detections_list[i]) > 0:
            CHW = torch.permute(torch.tensor(frame, dtype=torch.uint8), (2, 0, 1))  # move channels to front
            boxes_xyxy = box_convert(torch.Tensor(detections_list[i][:, 1:]), in_fmt="cxcywh", out_fmt="xyxy")
            drawn = draw_bounding_boxes(CHW, boxes_xyxy, colors="red", width=5)
            vid_writer.writeFrame(torch.permute(drawn, (1, 2, 0)).numpy())
        else:
            vid_writer.writeFrame(frame)

    vid_writer.close()
