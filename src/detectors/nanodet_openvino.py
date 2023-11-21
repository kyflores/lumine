import cv2
import numpy as np
import os
import detectors.nanodet_common as nc
import detectors.yolo_common as yc

from openvino.runtime import Core, Layout, get_batch


# nanodet.onnx from nanodet-plus-m_416
# Input: data (1, 3, 416, 416)
# Output output (1, 3598, 112) for COCO 80 classes
class NanodetOpenVinoDetector:
    def __init__(self, openvino_dir, backend="AUTO", dim=416):
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

        self.grid = nc.generate_grid_center_priors(dim, dim)

    def detect(self, im):
        im, self.scale = yc.resize_to_frame(im, self.dim)
        blob = cv2.dnn.blobFromImage(
            im,
            1.0 / 57.63,
            size=(im.shape[1], im.shape[0]),
            mean=(103.53, 116.28, 123.675),
            swapRB=True,
            crop=False,
        )
        y = list(self.executable_network([blob]).values())

        nms = nc.decode_infer(self.dim, self.dim, y[0], self.grid, 0.4)

        return yc.boxes_to_detection_dict(nms, self.dim, self.scale)
