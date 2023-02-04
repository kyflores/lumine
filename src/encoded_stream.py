# gstreamer backed videowriter for streaming compressed
# video to the driver station. H264 is preferred but H265 works
# with similar code.
#
# Note: default OpenCV from pip is not built with gstreamer, but
# at least in 22.04, the one in apt is.

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst


class EncodedStream:
    def __init__(
        self, resolution, bitrate=0, target_addr="127.0.0.1:51727", codec="H264"
    ):
        self.height, self.width = resolution
        self.bitrate = bitrate
        self.target_addr = target_addr

        pipeline = ""
        if codec == "H265":
            pipeline = """appsrc ! queue ! videoconvert !
                vaapih265enc bitrate={} ! h265parse !
                rtph265pay config-interval=1 ! udpsink clients={}"""
            pipeline = pipeline.format(bitrate, target_addr)
        elif codec == "H264":
            pipeline = """appsrc ! queue ! videoconvert !
                vaapih264enc bitrate={} ! h264parse !
                rtph264pay config-interval=1 ! udpsink clients={}"""
            pipeline = pipeline.format(bitrate, target_addr)
        else:
            print("Unknown Codec. Use H264 or H265")

        print(pipeline)

        self.stream = cv2.VideoWriter()

        self.stream.open(
            pipeline,
            cv2.CAP_GSTREAMER,  # FOURCC
            30,  # FPS
            (self.width, self.height),
            True,  # Color
        )

    def put_frame(self, frame):
        (rows, cols, _) = frame.shape
        assert self.width == cols
        assert self.height == rows

        self.stream.write(frame.astype(np.uint8))

    def destroy(self):
        self.stream.release()


if __name__ == "__main__":
    enc = EncodedStream((640, 480))

    import numpy as np
    import time

    while 1:
        enc.put_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        time.sleep(1 / 30)
        print("frame")
