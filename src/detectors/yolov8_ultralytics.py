import torch
import torchvision.ops as ops
import numpy as np
import cv2
import os
import time
from ultralytics import YOLO

import detectors.yolo_common as yc


class YoloUltralyticsDetector:
    def __init__(self, weights="yolov8n.pt", classes=yc.YOLOV5_CLASSES):
        self.classes = classes
        self.det = YOLO(weights)

    def detect(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.det([img])[
            0
        ]  # Returns as a list b/c this supports model input images at once
        res = []
        for b in results.boxes:
            for ids, conf, xyxy in zip(
                b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xyxy.cpu().numpy()
            ):
                # Order matters b/c of the script that draws them.
                corners = np.array(
                    (
                        (xyxy[0], xyxy[1]),
                        (xyxy[0], xyxy[3]),
                        (xyxy[2], xyxy[3]),
                        (xyxy[2], xyxy[1]),
                    )
                )
                res.append(
                    {
                        "type": "yolov8",
                        "id": ids,
                        "color": (0, 255, 0),
                        "corners": corners,
                        "confidence": conf,
                    }
                )
        return res
