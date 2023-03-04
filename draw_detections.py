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
    parser.add_argument("--conf_threshold", help="confidence threshold for removing uncertain predictions, must be in the range [0, 1].", default=0.0)
    args = parser.parse_args()
    name = args.name
    print("reading video")
    vid_generator = skvideo.io.vreader(f"io/{name}.mp4")
    vid_writer = skvideo.io.FFmpegWriter(f"io/{name}_detections.mp4")

    detections = np.load(f"internal/{name}.npz")
    detections_list = []
    for frame_number, frame_name in tqdm(enumerate(detections)):
        detections_list.append(detections[frame_name])

    for i, frame in tqdm(enumerate(vid_generator)):
        if len(detections_list[i]) > 0:
            CHW = torch.permute(torch.tensor(frame, dtype=torch.uint8), (2, 0, 1))  # move channels to front
            confs = detections_list[i][:, 0]
            detections_list[i] = detections_list[i][confs > float(args.conf_threshold)]
            boxes_xyxy = box_convert(torch.Tensor(detections_list[i][:, 1:]), in_fmt="cxcywh", out_fmt="xyxy")
            drawn = CHW
            for i, box in enumerate(boxes_xyxy):
                conf = confs[i]
                drawn = draw_bounding_boxes(CHW, box.unsqueeze(0), colors=(int(conf * 255), 255, 255), width=5)

            vid_writer.writeFrame(torch.permute(drawn, (1, 2, 0)).numpy())
        else:
            vid_writer.writeFrame(frame)

    vid_writer.close()
    print(f"finished writing io/{name}_detections.mp4")
