import cv2
import numpy as np
import detectors.yolo_common as yc
import os

from openvino.runtime import Core, Layout, get_batch


class YoloV8OpenVinoDetector:
    def __init__(
        self, openvino_dir, classes=yc.YOLOV5_CLASSES, backend="AUTO", dim=640
    ):
        self.dim = dim
        model = None
        weights = None
        meta = None
        mapping = None
        files = os.listdir(openvino_dir)
        for x in files:
            if x.endswith(".xml"):
                model = "{}/{}".format(openvino_dir, x)
            elif x.endswith(".bin"):
                weights = "{}/{}".format(openvino_dir, x)
            elif x.endswith(".yaml"):
                meta = "{}/{}".format(openvino_dir, x)
            elif x.endswith(".mapping"):
                mapping = "{}/{}".format(openvino_dir, x)

        self.classes = classes
        self.scale = 1.0

        self.ie = Core()
        self.network = self.ie.read_model(model=model, weights=weights)

        if self.network.get_parameters()[0].get_layout().empty:
            self.network.get_parameters()[0].set_layout(Layout("NCHW"))

        batch_dim = get_batch(self.network)
        if batch_dim.is_static:
            batch_size = batch_dim.get_length()
        self.executable_network = self.ie.compile_model(
            self.network, device_name=backend
        )

    def detect(self, im):  # img is a np array
        im, self.scale = yc.resize_to_frame(im, self.dim)
        blob = cv2.dnn.blobFromImage(
            im,
            1.0 / 255,
            size=(im.shape[1], im.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )

        y = list(self.executable_network([blob]).values())

        nms = yc.process_yolov8_output_tensor(y[0])
        return yc.boxes_to_detection_dict(nms, self.dim, self.scale)
