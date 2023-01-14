import torch
import torchvision.ops as ops
import numpy as np
import cv2
import os
import time

# TODO: For OpenCV DNN
BOX_THRESH = 0.25
CLASS_THRESH = 0.25
NMS_THRESH = 0.45
NMS_SCORE_THRESH = BOX_THRESH * CLASS_THRESH

# Relative to main.py
YOLOV5_PATH = os.path.dirname(__file__) + "/../subprojects/yolov5/"
# Classes in the yolov5 model. Indexes should match the ultralytics/yolov5 pretrained models.
YOLOV5_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# owh is Origin, Width, Height. OpenCV uses this format
def xywh2owh(x):
    x[:, 0] = x[:, 0] - x[:, 2] / 2  # x origin
    x[:, 1] = x[:, 1] - x[:, 3] / 2  # y origin
    return x

def process_yolo_output_tensor(tensor):
    tensor = tensor.squeeze()

    best_score = np.max(tensor[:, 5:], axis=1)
    # Where the best score >= the class thresh
    valid = best_score >= CLASS_THRESH
    tensor = tensor[valid]

    class_ids = np.argmax(tensor[:, 5:], axis=1)
    boxes = xywh2owh(tensor[:, :4])
    confidences = tensor[:, 4:5].squeeze() * best_score[valid]

    nms_res = cv2.dnn.NMSBoxes(
        boxes, confidences, NMS_SCORE_THRESH, NMS_THRESH
    )

    return (
        nms_res,
        boxes,
        confidences,
        class_ids,
    )

class YoloV5TorchDetector:
    def __init__(self, weights="yolov5s.pt", classes=YOLOV5_CLASSES):
        self.classes = classes
        self.det = torch.hub.load(YOLOV5_PATH, "custom", weights, source="local")

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

# Uses OpenCV dnn for inference. The model must be exported
# to ONNX for this backend.
#
# The additional inference backends must be enabled in OpenCV
# at compile time. IIRC opencv-python build on PyPi does NOT enable
# cuda, vulkan, or openvino. When compiling OpenCV, it's worth trying
# to include MKL as well, to improve the CPU backend performance.
class YoloV5OpenCVDetector:
    def __init__(self, weights="yolov5s.onnx", classes=YOLOV5_CLASSES, backend="cpu"):
        self.classes = classes
        self.net = cv2.dnn.readNet(weights)
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
        num_boxes = 25200
        self.class_ids = np.empty(num_boxes)
        self.confidences = np.empty(num_boxes)
        self.boxes = np.empty((num_boxes, 4))

    def detect(self, img):
        img = self._resize_to_frame(img)
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / 255,
            size=(img.shape[1], img.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )

        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        (nms_res, boxes, confidences, class_ids) = process_yolo_output_tensor(outs[0])
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
                    "id": classnm,
                    "color": (0, 255, 0),
                    "corners": corners,
                    "confidence": conf,
                }
            )
        return res

    def _resize_to_frame(self, imraw):
        major_dim = np.max(imraw.shape)
        scale = 640 / major_dim
        imraw = cv2.resize(imraw, None, fx=scale, fy=scale)
        img = np.zeros((640, 640, 3), dtype=imraw.dtype)
        img[: imraw.shape[0], : imraw.shape[1], :] = imraw
        return img


from openvino.runtime import Core, Layout, get_batch

class YoloV5OpenVinoDetector:
    def __init__(self, openvino_dir, classes=YOLOV5_CLASSES, backend="CPU"):
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
        self.class_ids = None
        self.confidences = None
        self.boxes = None

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
        im = self.resize_to_frame(im)
        blob = cv2.dnn.blobFromImage(
            im,
            1.0 / 255,
            size=(im.shape[1], im.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )

        y = list(self.executable_network([blob]).values())

        (nms_res, boxes, confidences, class_ids) = process_yolo_output_tensor(y[0])

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
                    "id": self.classes[classnm],
                    "color": (0, 255, 0),
                    "corners": corners,
                    "confidence": conf,
                }
            )
        return res

    def resize_to_frame(self, imraw):
        if imraw.shape == (640, 640, 3):
            return imraw

        major_dim = np.max(imraw.shape)
        scale = 640 / major_dim
        imraw = cv2.resize(imraw, None, fx=scale, fy=scale)
        img = np.zeros((640, 640, 3), dtype=imraw.dtype)
        img[: imraw.shape[0], : imraw.shape[1], :] = imraw
        return img
