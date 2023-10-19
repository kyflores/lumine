import argparse
import cv2
import numpy as np
import os
import random
from tqdm import tqdm

import matplotlib.pyplot as plt

# Based on estimates of opencv's 0->180 colorwheel
YELLOW = (20, 30)
PURPLE = (125, 140)


def random_hue(color="yellow"):
    if color == "yellow":
        return random.randint(YELLOW[0], YELLOW[1])
    elif color == "purple":
        return random.randint(PURPLE[0], PURPLE[1])
    else:
        choice = random.choice([YELLOW, PURPLE])
        return random.randint(choice[0], choice[1])


def color_pixels(im, new_hue, mode="box", grey=0.0, debug=False):
    scr = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    rows, cols, chans = scr.shape
    rand_x = np.random.randint(0, cols)
    rand_y = np.random.randint(0, rows)
    point = (rand_x, rand_y)

    if mode == "bucket":
        l_thresh = random.randint(5, 20)
        u_thresh = random.randint(5, 20)
        ret, recolored, mask, rect = cv2.floodFill(
            scr,
            None,
            seedPoint=point,  # (pair)
            newVal=(100, 100, 100),
            upDiff=(u_thresh, u_thresh // 2, u_thresh // 2),  # R G B
            loDiff=(l_thresh, l_thresh // 2, l_thresh // 2),
        )

        mask = mask[1:-1, 1:-1]

        as_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        as_hsv[:, :, 0] = np.where(mask, new_hue, as_hsv[:, :, 0])
        return cv2.cvtColor(as_hsv, cv2.COLOR_HSV2BGR), rect
    elif mode == "box":
        tmp = np.min((rows, cols))

        # Make boxes between 10% and 40% the smaller of the image dimensions.
        min_dim = int(tmp * 0.10)
        max_dim = int(tmp * 0.40)
        w = random.randint(min_dim, max_dim)
        h = random.randint(min_dim, max_dim)

        mask = np.zeros((rows, cols))
        mask[rand_y : rand_y + h, rand_x : rand_x + w] = 1

        value = random.randint(120, 230)
        saturation = random.randint(150, 250)
        as_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        as_hsv[..., 0] = np.where(mask, new_hue, as_hsv[..., 0])
        as_hsv[..., 1] = np.where(mask, saturation, as_hsv[..., 1])
        as_hsv[..., 2] = np.where(mask, value, as_hsv[..., 2])
        return cv2.cvtColor(as_hsv, cv2.COLOR_HSV2BGR), (
            rand_x,
            rand_y,
            rand_x + w,
            rand_y + h,
        )


def random_grey(im, grey=0.0):
    as_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    if random.uniform(0.0, 1.0) < grey:
        as_hsv[..., 1] = 0

    return cv2.cvtColor(as_hsv, cv2.COLOR_HSV2BGR)


def paint_images(im_path, label_path):
    im_names = os.listdir(im_path)
    lbl_names = os.listdir(label_path)

    im_names.sort()
    lbl_names.sort()

    # Basic check that label and images directories match.
    assert len(im_names) == len(lbl_names)

    os.makedirs(os.path.join(im_path, "rc"), exist_ok=True)
    for img_name, lbl_name in tqdm(zip(im_names, lbl_names)):
        im = cv2.imread(os.path.join(im_path, img_name))
        if im is None:
            print("Read null image", img_name)
            continue

        rows, cols, channels = im.shape

        for x in range(random.randint(0, 4)):
            im, _rect = color_pixels(im, random_hue("neither"))

        im = random_grey(im, 0.5)
        cv2.imwrite(os.path.join(im_path, "rc", img_name), im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgdir",
        type=str,
        help="Directory of images to recolor",
    )

    # UNIMPLEMENTED
    parser.add_argument(
        "--labeldir",
        type=str,
        help="Directory of labels matching the images. If passed, recolors do not intersect labels",
    )

    opt = parser.parse_args()

    paint_images(opt.imgdir, None)
