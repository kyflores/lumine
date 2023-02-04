import cv2
import numpy as np
import detectors.yolo_common as yc
import os

from openvino.runtime import Core, Layout, get_batch


class YoloV5OpenVinoDetector:
    def __init__(self, openvino_dir, classes=yc.YOLOV5_CLASSES, backend="AUTO"):
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
        im, self.scale = yc.resize_to_frame(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(
            im,
            1.0 / 255,
            size=(im.shape[1], im.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )

        y = list(self.executable_network([blob]).values())

        (nms_res, boxes, confidences, class_ids) = yc.process_yolo_output_tensor(y[0])

        res = []
        for idx in nms_res:
            conf = confidences[idx]
            classnm = class_ids[idx]
            x, y, w, h = np.clip(boxes[idx], 0, 640).astype(np.uint32)
            d = (x, y, x + w, y + h)  # xyxy format
            corners = np.array(((d[0], d[1]), (d[0], d[3]), (d[2], d[3]), (d[2], d[1])))

            res.append(
                {
                    "type": "yolov5",
                    # "id": self.classes[classnm], # Numbers are more convenient for NT.
                    "id": classnm,
                    "color": (0, 255, 0),
                    "corners": corners * self.scale,
                    "confidence": conf,
                }
            )
        return res
