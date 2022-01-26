import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from torchvision.io import read_image
import numpy as np


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


def detect_single_frame(image):

    model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91,
                                    pretrained_backbone=True)
    model.eval()
    detections = model(x)[0]
    print(detections)
    detections["boxes"] = detections["boxes"][detections["labels"] == 37]
    return detections

if __name__ == "__main__":
    image = Image.open("test_balls.jpg")
    x = F.to_tensor(image)
    x.unsqueeze_(0)
    print(x.shape)

    res = detect_single_frame(x)
    show(draw_bounding_boxes(read_image("test_balls.jpg"), res["boxes"], width=10, colors="red"))
    plt.show()

