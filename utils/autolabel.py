# Use a yolov5 network to label a dataset dump it to the yolo format.
# The output directory is formatted to match the "YOLO 1.1" format that
# CVAT expects, and becomes an image sequence rather than a video dataset

# Top
#   obj.data
#   obj.names
#   train.txt
#   obj_train_data/
#     imgXXX.png
#     imgXXX.txt

import os
import numpy as np
import cv2
import torch

import argparse
import shutil

YOLOV5_PATH = os.path.dirname(__file__) + "/../src/subprojects/yolov5/"


class YoloV5Wrapper:
    def __init__(self, weights="yolov5s.pt"):
        self.model = torch.hub.load(YOLOV5_PATH, "custom", weights, source="local")
        self.classes = self.model.names

    def metadata(self):
        idxs = list(
            self.classes.items()
        )  # Returns like (0, person), (1, bicycle), but maybe not in order
        idxs.sort(key=lambda x: x[0])
        class_names = [x[1] for x in idxs]

        return len(class_names), class_names

    def detect(self, img, min_conf=0.4):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detects = self.model.forward(img).xywh[0].cpu().numpy()
        res = []

        height, width, channel = img.shape

        for d in detects:
            x, y, w, h, conf, classnm = d[:6]

            x = x / width
            y = y / height
            w = w / width
            h = h / height

            res.append((int(classnm), x, y, w, h))

        return res


class DatasetWriter:
    def __init__(self, model):
        self.root = "data"
        self.objdir = os.path.join(self.root, "obj_train_data")

        self.frame_counter = 0
        self.model = model

        num_classes, class_nms = self.model.metadata()

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.objdir, exist_ok=True)

        fname = os.path.join(self.root, "obj.data")
        with open(fname, "w") as data:
            lines = [
                "classes = {}".format(int(num_classes)),
                "train = data/train.txt",
                "names = data/obj.names",
                "backup = backup/",
            ]
            for l in lines:
                data.write(l + "\n")

        fname = os.path.join(self.root, "obj.names")
        with open(fname, "w") as names:
            for l in class_nms:
                names.write(l + "\n")

        # Leave this one open because each image appears here.
        fname = os.path.join(self.root, "train.txt")
        self.train_file_handle = open(fname, "w")

    def add_frame(self, frame):
        boxes = self.model.detect(frame, 0.0)
        base_name = "frame_" + str(self.frame_counter).zfill(6)
        self.frame_counter += 1

        if len(boxes) == 0:
            print("Frame {}: Found no boxes. Skipping...".format(self.frame_counter))
            return
        else:
            print(
                "Frame {}: generated {} detects".format(self.frame_counter, len(boxes))
            )

        label_name = os.path.join(self.root, "obj_train_data", base_name + ".txt")
        img_name = os.path.join(self.root, "obj_train_data", base_name + ".png")
        with open(label_name, "w") as labelfile:
            for box in boxes:
                line = " ".join(str(x) for x in box)
                labelfile.write(line + "\n")

        cv2.imwrite(img_name, frame)

        self.train_file_handle.write(img_name + "\n")

    def cleanup(self):
        self.train_file_handle.close()


class VideoLoader:
    def __init__(self, videopath, every_n=1):
        self.reader = cv2.VideoCapture(videopath)
        self.every_n = every_n
        if not self.reader.isOpened():
            raise Exception(
                "Could not open video file source. Maybe bad path or incompatible format."
            )

    def next(self):
        frame = None
        for _ in range(self.every_n):
            # We read N frames but only return the last one. Naive use of temporal codecs requires
            # decoding of every frame sequentially anyway, which includes just about any
            # recent video.
            _, frame = self.reader.read()
            if frame is None:
                return None

        return frame

    def __del__(self):
        self.reader.release()


def main(opt):
    model = YoloV5Wrapper(weights=opt.weights)
    writer = DatasetWriter(model)
    loader = VideoLoader(opt.video, every_n=opt.decimate)

    fr = loader.next()
    while fr is not None:
        writer.add_frame(fr)
        fr = loader.next()
    writer.cleanup()

    print("Inference is complete, zipping up your dataset...")
    shutil.make_archive("lumine_autolabel_out", "zip", writer.root)

    print("Done! Select YOLO 1.1 as the format when importing into cvat.")
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video to label, any format supported by cv2.VideoCapture",
    )
    parser.add_argument(
        "--weights", type=str, help="Path to custom weights, eg best.pt"
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=1,
        help="Use every nth frame. eg --decimate=4 only analyzes and saves every 4th frame.",
    )
    opt = parser.parse_args()

    main(opt)
