# Use OWLViT to label a dataset and dump it to the yolo format.
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

from transformers import OwlViTProcessor, OwlViTForObjectDetection
# from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers.image_utils import ImageFeatureExtractionMixin

import argparse
import shutil

# Based on https://github.com/huggingface/notebooks/blob/main/examples/zeroshot_object_detection_with_owlvit.ipynb


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

class OWLViT:
    def __init__(self, model_name="google/owlvit-base-patch32") -> None:
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.mixin = ImageFeatureExtractionMixin()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval()
        self.model.to(self.device)

    def detect(self, text_queries, image):
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        image_size = self.model.config.vision_config.image_size
        image_tmp = self.mixin.resize(image, image_size)
        input_image = np.asarray(image_tmp).astype(np.float32) / 255.0

        # Threshold to eliminate low probability predictions
        score_threshold = 0.1

        # Get prediction logits
        logits = torch.max(outputs["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().detach().numpy()

        # Get prediction labels and boundary boxes
        labels = logits.indices.cpu().detach().numpy()
        boxes = outputs["pred_boxes"][0].cpu().detach().numpy()
        for score, box, label in zip(scores, boxes, labels):
            if score < score_threshold:
                continue

            cx, cy, w, h = box
            print(score, box)


def main(opt):
    owl = OWLViT()
    # writer = DatasetWriter(model)
    loader = VideoLoader(opt.video, every_n=opt.decimate)

    fr = loader.next()
    while fr is not None:
        # writer.add_frame(fr)
        owl.detect("orange ring", fr)
        fr = loader.next()
    # writer.cleanup()

    # print("Inference is complete, zipping up your dataset...")
    # shutil.make_archive("lumine_autolabel_out", "zip", writer.root)

    # print("Done! Select YOLO 1.1 as the format when importing into cvat.")
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
