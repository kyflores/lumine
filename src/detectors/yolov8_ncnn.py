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
        blob = cv2.dnn.blobFromImage(
            im,
            1.0 / 255,
            size=(im.shape[1], im.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        ex = self.net.create_extractor()
        ex.input("in0", ncnn.Mat(blob).clone())
        t_b = time.time()

        _, result_tensor = ex.extract("out0")
        result_tensor = np.expand_dims(np.array(result_tensor), axis=0)
        print(result_tensor.shape)

        t_e = time.time()

        nms = yc.process_yolov8_output_tensor(result_tensor)

        return yc.boxes_to_detection_dict(nms, self.dim, self.scale)
