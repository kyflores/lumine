import cv2
import numpy as np
import detectors.yolo_common as yc
import os

from deepsparse import compile_model


class YoloV8DeepsparseDetector:
    def __init__(self, dim=640):
        self.dim = dim
        self.stub = "zoo:yolov8-n-coco-pruned49_quantized"
        self.compiled_model = compile_model(self.stub, batch_size=1)
        print(self.compiled_model)

    def detect(self, im):  # img is a np array
        im, self.scale = yc.resize_to_frame(im, self.dim)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        im = np.moveaxis(im, 2, 0)
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im)

        y = self.compiled_model([im])

        nms = yc.process_yolov8_output_tensor(y[0])
        return yc.boxes_to_detection_dict(nms, self.dim, self.scale)
