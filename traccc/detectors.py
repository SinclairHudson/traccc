"""Implements various detectors for object detection in videos."""
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.functional as F
from torchvision.io import write_video
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_convert
from torchvision.transforms import Normalize
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
from transformers import DetrFeatureExtractor, DetrForObjectDetection, pipeline
from typing import List
from PIL import Image

SPORTS_BALL_COCO_CLASS_IDX = 37

def show(imgs):
    """
    Helper from pytorch tutorials.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)  # pylint: disable=no-member
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


class Detector(ABC):
    def __init__(self):
        raise NotImplementedError

    def detect_video(self, video, bbox_format="cxcywh", frame_count: int = None, prompts: List[str] = None):
        raise NotImplementedError

    def detect(self, video, filename="internal/detections.npz",
               frame_count: int = None,
               prompts: List[str] = None,
               progress=None):
        if prompts is None:
            detections = self.detect_video(video, frame_count=frame_count)
        else:
            detections = self.detect_video(video, frame_count=frame_count, prompts=prompts)
        np.savez(filename, *detections)
        return f"Successfully saved detections in {filename}"

    def display_detections_in_video(self, video: torch.Tensor, outfile: str) -> None:
        detections = self.detect_video(video, bbox_format="xyxy")
        print("drawing bounding boxes")
        for i, frame in tqdm(enumerate(video)):
            # draw boxes on single frame
            # write frame to mp4
            CHW = torch.permute(frame, (2, 0, 1))  # move channels to front
            video[i] = torch.permute(draw_bounding_boxes(CHW, torch.Tensor(
                detections[i]), colors="red", width=5), (1, 2, 0))  # move C back to end, save in tensor
        print("writing the video")
        write_video(outfile, video, video_codec="h264", fps=60.0)


class PretrainedRN50Detector(Detector):
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91,
                                             pretrained_backbone=True)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(self.device)

    @torch.no_grad()
    def detect_video(self, video, bbox_format="cxcywh", frame_count: int = None, prompts: str = None):
        """
        video is a generator
        output is a list of numpy arrays denoting bounding boxes for each frame
        """
        # TODO batch these calls
        video_detections = []  # list of list of detections
        ZeroOne = Normalize((0, 0, 0), (255, 255, 255))  # divide to 0 to 1
        # num_batches = len(video) // batch_size  # last frames may be cut

        print("detecting balls in the video")
        # for frame in tqdm(video, total=frame_count):
        for frame, _ in zip(video, tqdm(range(frame_count))):
            batch = torch.Tensor(frame).unsqueeze(0)
            batch = torch.moveaxis(batch, 3, 1)  # move channels to position 1
            batch = ZeroOne(batch.float())
            batched_result = self.model(batch.to(self.device))
            for res in batched_result:
                xyxy = res["boxes"][res["labels"] == SPORTS_BALL_COCO_CLASS_IDX]
                conf = res["scores"][res["labels"] == SPORTS_BALL_COCO_CLASS_IDX]
                xywh = box_convert(xyxy, in_fmt="xyxy", out_fmt=bbox_format)
                cxywh = torch.cat((conf.unsqueeze(1), xywh),
                                  dim=1)  # add confidences
                video_detections.append(cxywh.cpu().numpy())
        return video_detections


class HuggingFaceDETR(Detector):
    def __init__(self):
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(
            'facebook/detr-resnet-101-dc5')
        self.model = DetrForObjectDetection.from_pretrained(
            'facebook/detr-resnet-101-dc5')

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(self.device)

    @torch.no_grad()
    def detect_video(self, video, bbox_format="cxcywh", frame_count=None, prompts: str = None):
        """
        video is a generator
        output is a list of numpy arrays denoting bounding boxes for each frame
        """
        video_detections = []  # list of list of detections
        # num_batches = len(video) // batch_size  # last frames may be cut

        print("detecting balls in the video")
        # for frame in tqdm(video, total=frame_count):
        for frame, _ in zip(video, tqdm(range(frame_count))):
            width, height, _ = frame.shape
            inputs = self.feature_extractor(images=frame, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
            inputs["pixel_mask"] = inputs["pixel_mask"].to(self.device)

            outputs = self.model(**inputs)
            outputs["logits"] = outputs["logits"].squeeze(
                0)  # remove singleton batch dim
            outputs["pred_boxes"] = outputs["pred_boxes"].squeeze(0)
            confs = torch.nn.functional.softmax(outputs["logits"], dim=1)
            conf_scores, indices = torch.max(confs, dim=1)
            xywh = outputs["pred_boxes"][indices == SPORTS_BALL_COCO_CLASS_IDX]
            conf_scores = conf_scores[indices == SPORTS_BALL_COCO_CLASS_IDX]

            xywh[:, 0] *= height
            xywh[:, 2] *= height
            xywh[:, 1] *= width
            xywh[:, 3] *= width
            cxywh = torch.cat((conf_scores.unsqueeze(1), xywh),
                              dim=1)  # add confidences
            video_detections.append(cxywh.cpu().numpy())
        return video_detections


class OWLVITZeroShot(Detector):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = pipeline(model="google/owlvit-base-patch32",
                                          task="zero-shot-object-detection", device=self.device)


    @torch.no_grad()
    def detect_video(self, video, prompts: List[str], bbox_format="cxcywh", frame_count=None):
        """
        Args:
            video: a generator that goes through frames of the video.
            prompts: list of strings describing objects.
            bbox_format: format of the bounding boxes, either "cxcywh" or "xyxy".
        """
        video_detections = []  # list of list of detections
        # num_batches = len(video) // batch_size  # last frames may be cut

        print("detecting balls in the video")
        print(self.device)
        for frame in tqdm(video, total=frame_count):

            pipeline_output = self.feature_extractor(image=Image.fromarray(frame), candidate_labels=prompts)
            scores = []
            xs = []
            ys = []
            widths = []
            heights = []
            for detection in pipeline_output:
                scores.append(detection["score"])
                w = detection["box"]["xmax"] - detection["box"]["xmin"]
                h = detection["box"]["ymax"] - detection["box"]["ymin"]
                xs.append(detection["box"]["xmin"] + w / 2)
                ys.append(detection["box"]["ymin"] + h / 2)
                widths.append(w)
                heights.append(h)
            cxywh = np.stack((scores, xs, ys, widths, heights), axis=1)
            video_detections.append(cxywh)
        return video_detections
