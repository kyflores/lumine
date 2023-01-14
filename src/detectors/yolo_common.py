# Common functions for YoloV5
import numpy as np
import cv2
import os

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

    nms_res = cv2.dnn.NMSBoxes(boxes, confidences, NMS_SCORE_THRESH, NMS_THRESH)

    return (
        nms_res,
        boxes,
        confidences,
        class_ids,
    )
