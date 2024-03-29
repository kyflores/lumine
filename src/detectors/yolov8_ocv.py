import cv2
import numpy as np
import detectors.yolo_common as yc


# Uses OpenCV dnn for inference. The model must be exported
# to ONNX for this backend.
#
# The additional inference backends must be enabled in OpenCV
# at compile time. IIRC opencv-python build on PyPi does NOT enable
# cuda, vulkan, or openvino. When compiling OpenCV, it's worth trying
# to include MKL as well, to improve the CPU backend performance.
#
# Baseline CPU performance isn't great, so recommended to use this only as
# a testing aid. The CPU backend doesn't seem SMT aware as it still maxes out
# the logical cores.
class YoloV8OpenCVDetector:
    def __init__(
        self, weights="yolov8n.onnx", dim=640, classes=yc.YOLOV5_CLASSES, backend="cpu"
    ):
        self.dim = dim
        self.classes = classes
        self.net = cv2.dnn.readNet(weights)
        self.scale = 1.0
        if backend == "vulkan":
            print("Vulkan will be used if it is available.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_VKCOM)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_VULKAN)
        elif backend == "opencl":
            print("OpenCL will be used if it is available.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        elif backend == "cuda":
            print("CUDA will be used if it is available.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        elif backend == "cpu":
            print("CPU backend will be used.")
        else:
            print("Unknown backend, CPU backend will be used.")

        self.out_names = self.net.getUnconnectedOutLayersNames()

    def detect(self, img):
        img, self.scale = yc.resize_to_frame(img, self.dim)
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / 255,
            size=(img.shape[1], img.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        nms = yc.process_yolov8_output_tensor(outs[0])
        return yc.boxes_to_detection_dict(nms, self.dim, self.scale)
