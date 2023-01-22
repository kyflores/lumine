#! Wrapper class for writing arbitrary images to the dashboard.

from cscore import CameraServer
import cv2
import numpy as np


class CsCoreStream:
    def __init__(self, shape, name):
        assert len(shape) == 2
        self.cs = CameraServer.getInstance()
        self.cs.enableLogging()

        self.rows, self.cols = shape
        self.ostream = self.cs.putVideo(name, self.cols, self.rows)

    def write_frame(self, fr):
        rows, cols, channels = fr.shape
        assert channels == 3

        if (rows != self.rows) or (cols != self.cols):
            fr = cv2.resize(fr, (cols, rows))

        self.ostream.putFrame(fr)

    # TODO, need to release the stream?
    # def destroy
