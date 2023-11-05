import cv2
import numpy as np
import detectors.yolo_common as yc
import os
import onnx
import time
from ncnn_vulkan import ncnn


# pip install ncnn-vulkan
class YoloV8NcnnDetector:
    def __init__(
        self,
        model_dir,
        dim=640,
    ):
        self.params_nm = os.path.join(model_dir, "model.ncnn.param")
        self.model_nm = os.path.join(model_dir, "model.ncnn.bin")
        self.dim = dim

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = True
        self.net.load_param(self.params_nm)
        self.net.load_model(self.model_nm)

    def detect(self, im):
        im, self.scale = yc.resize_to_frame(im, self.dim)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(
            im,
            1.0 / 255,
            size=(im.shape[1], im.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )
        ex = self.net.create_extractor()
        ex.input("in0", ncnn.Mat(blob).clone())
        t_b = time.time()

        _, result_tensor = ex.extract("out0")
        result_tensor = np.expand_dims(np.array(result_tensor), axis=0)
        print(result_tensor.shape)

        t_e = time.time()

        (nms_res, boxes, confidences, class_ids) = yc.process_yolov8_output_tensor(
            result_tensor
        )

        res = []
        for idx in nms_res:
            conf = confidences[idx]
            classnm = class_ids[idx]
            x, y, w, h = np.clip(boxes[idx], 0, self.dim).astype(np.uint32)
            d = (x, y, x + w, y + h)  # xyxy format
            corners = np.array(((d[0], d[1]), (d[0], d[3]), (d[2], d[3]), (d[2], d[1])))

            res.append(
                {
                    "type": "yolov8",
                    "id": classnm,
                    "color": (0, 0, 255),
                    "corners": corners * self.scale,
                    "confidence": conf,
                }
            )
        return res
