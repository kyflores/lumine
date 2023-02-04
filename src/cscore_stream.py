#! Wrapper class for writing arbitrary images to the dashboard.

import ntcore
import cscore
import cv2
import time
import socket


class CsCoreStream:
    def __init__(self, shape, ip, port, fps=30):
        assert len(shape) == 2
        print("Setting up Cscore")
        self.rows, self.cols = shape
        self.src = cscore.CvSource(
            "lumine_src", cscore.VideoMode.PixelFormat.kMJPEG, self.cols, self.rows, fps
        )
        self.srv = cscore.MjpegServer("lumine_srv", int(port))
        self.srv.setSource(self.src)
        self.nt = ntcore.NetworkTableInstance.getDefault()
        self.stream_uri = (
            self.nt.getTable("CameraPublisher/lumine")
            .getStringArrayTopic("streams")
            .publish()
        )

        self.stream_uri.set(["mjpeg:http://{}:{}/?action=stream".format(ip, port)])

        self.last = time.time()

    def write_frame(self, fr):
        rows, cols, channels = fr.shape
        assert channels == 3

        if (rows != self.rows) or (cols != self.cols):
            fr = cv2.resize(fr, (self.cols, self.rows))

        self.src.putFrame(fr)

    # TODO, need to release the stream?
    # def destroy
