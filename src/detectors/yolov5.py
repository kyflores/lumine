import torch
import torchvision.ops as ops
import numpy as np
import cv2
import os
import time

import detectors.yolo_common as yc


class YoloV5TorchDetector:
    def __init__(self, weights="yolov5s.pt", classes=yc.YOLOV5_CLASSES):
        self.classes = classes
        self.det = torch.hub.load(yc.YOLOV5_PATH, "custom", weights, source="local")

    def detect(self, img):
        detects = self.det.forward(img).xyxy[0].cpu().numpy()
        res = []
        for d in detects:
            # Order matters b/c of the script that draws them.
            corners = np.array(((d[0], d[1]), (d[0], d[3]), (d[2], d[3]), (d[2], d[1])))
            conf = d[4]
            classnm = self.classes[int(d[5])]
            res.append(
                {
                    "type": "yolov5",
                    "id": classnm,
                    "color": (0, 255, 0),
                    "corners": corners,
                    "confidence": conf,
                }
            )
        return res
