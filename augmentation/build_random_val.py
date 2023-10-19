import sys
import os
import random
import shutil

if __name__ == "__main__":
    train = sys.argv[1]
    val = sys.argv[2]
    images = "images"
    labels = "labels"

    files = os.listdir(os.path.join(train, images))
    num_images = len(files)

    choices = random.choices(files, k=num_images // 10)

    for c in choices:
        basename = os.path.basename(c)
        name_only = os.path.splitext(basename)[0]
        label_name = name_only + ".txt"
        shutil.copyfile(os.path.join(train, images, c), os.path.join(val, images, c))

        shutil.copyfile(
            os.path.join(train, labels, label_name),
            os.path.join(val, labels, label_name),
        )
