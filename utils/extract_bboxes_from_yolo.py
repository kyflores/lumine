# This script goes through a dataset in the yolov5 format and extracts
# each bounding box into its own image.
# The format is extremely simple: Each image is a png whose name is X_YYYYYY.png,
# where X is the class number, and YYYYYY is a monotonically increasing value
# to guarantee that each image has a unique name.

import cv2
import os
import sys
import numpy as np

def resize_to_frame(imraw, dim):
    major_dim = np.max(imraw.shape)
    scale = dim / major_dim
    outscale = 1 / scale
    imraw = cv2.resize(imraw, None, fx=scale, fy=scale)
    img = np.zeros((dim, dim, 3), dtype=imraw.dtype)
    img[: imraw.shape[0], : imraw.shape[1], :] = imraw
    return img, outscale

def main(datadir, target_size):
    os.makedirs("classifier_dataset", exist_ok=True)
    imgdir = os.path.join(datadir, "images")
    lbldir = os.path.join(datadir, "labels")

    count = 0;
    label_files = os.listdir(lbldir)
    for lbl in label_files:
        base = os.path.basename(lbl)
        (name, ext) = os.path.splitext(base)

        img = cv2.imread(os.path.join(imgdir, name + ".PNG"))
        (rows, cols, channels) = img.shape

        boxes = []
        with open(os.path.join(lbldir,lbl), 'r') as lf:
            for line in lf:
                vals = line.split(' ')

                x = float(vals[1]) * cols
                y = float(vals[2]) * rows
                w = float(vals[3]) * cols
                h = float(vals[4]) * rows

                xmin = int(x - w // 2)
                xmax = int(x + w // 2)
                ymin = int(y - h // 2)
                ymax = int(y + h // 2)

                boxes.append({
                    "cls": int(vals[0]),
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax
                })

        for box in boxes:
            imgname = os.path.join(
                "classifier_dataset",
                "{}_{}.PNG".format(box["cls"], str(count).zfill(6))
            )
            im = img[box["ymin"]:box["ymax"], box["xmin"]:box["xmax"], :]
            rsz, _ = resize_to_frame(im, target_size)
            cv2.imwrite(imgname, rsz)

            count += 1



if __name__ == '__main__':
    target = 28 # 28x28 is mnist default dimensions.
    datadir = sys.argv[1]
    main(datadir, target)
