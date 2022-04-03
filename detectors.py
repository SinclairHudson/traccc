from abc import ABC
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.io import write_video
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToPILImage, ToTensor, Normalize


SPORTS_BALL = 37  # from coco class mapping


def show(imgs):
    """
    Helper from pytorch tutorials
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


class Detector(ABC):
    def __init__(self):
        raise NotImplementedError

    def detect_video(self, video, bbox_format="cxcywh"):
        raise NotImplementedError

    def detect(self, video, filename="io/detections.npz"):
        detections = self.detect_video(video)
        np.savez(filename, *detections)
        print(f"saved detections in {filename}")

    def display_detections_in_video(self, video: torch.Tensor, outfile: str) -> None:
        detections = self.detect_video(video, bbox_format="xyxy")
        print("drawing bounding boxes")
        for i, frame in tqdm(enumerate(video)):
            # draw boxes on single frame
            # write frame to mp4
            CHW = torch.permute(frame, (2, 0, 1))  # move channels to front
            video[i] = torch.permute(draw_bounding_boxes(CHW, torch.Tensor(detections[i]), colors="red", width=5), (1, 2, 0))  # move C back to end, save in tensor
        print("writing the video")
        write_video(outfile, video, video_codec="h264", fps=60.0)


class RawPretrainedDetector(Detector):
    def __init__(self, device='cpu'):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91,
                                        pretrained_backbone=True)

        self.device = torch.device(device)
        self.model.eval().to(self.device)


    @torch.no_grad()
    def detect_video(self, video: torch.Tensor, bbox_format="cxcywh", conf_threshold=0.5):
        """
        video is a sequence of frames THWC (time, height, width, channel), pytorch tensor
        output is a list of numpy arrays denoting bounding boxes for each frame
        """
        batch_size = 16
        num_batches = len(video) // batch_size  # last frames may be cut
        video_detections = []  # list of list of detections
        TPI = ToPILImage()
        TT = ToTensor()
        ZeroOne = Normalize((0, 0, 0), (255, 255, 255))  # divide to 0 to 1
        print("detecting balls in the video")
        for index in tqdm(range(num_batches)):
            batch = video[index * batch_size : (index + 1) * batch_size]
            batch = torch.moveaxis(batch, 3, 1)  # move channels to position 1
            batch = ZeroOne(batch.float())
            batched_result = self.model(batch.to(self.device))
            for res in batched_result:
                xyxy = res["boxes"][res["labels"] == SPORTS_BALL]
                xywh = box_convert(xyxy, in_fmt="xyxy", out_fmt=bbox_format)
                video_detections.append(xywh.cpu().numpy())

        if len(video) % batch_size != 0:  # there's leftover
            batch = video[num_batches * batch_size:]
            batch = torch.moveaxis(batch, 3, 1)  # move channels to position 1
            batch = ZeroOne(batch.float())
            batched_result = self.model(batch.to(self.device))
            for res in batched_result:
                xyxy = res["boxes"][torch.logical_and(res["labels"] == SPORTS_BALL, res["scores"] > conf_threshold)]
                # TODO add some sort of confidence threshold.
                xywh = box_convert(xyxy, in_fmt="xyxy", out_fmt=bbox_format)
                video_detections.append(xywh.cpu().numpy())

        assert len(video_detections) == len(video)
        return video_detections




