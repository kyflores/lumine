import numpy as np

# Dummy detector class that demonstrates the minimal detector API
# Draws a box that marches diagonally across the frame one pixel at a time.
class DummyDetector:
    def __init__(self, size=16):
        self.size = size
        self.idx = 0

    def detect(self, img):
        corners = self.generate_box_corners(img.shape)

        res = []
        res.append(
            {
                "type": "dummy",
                "id": "box",
                "color": (255, 255, 255),
                "confidence": 1,
                "corners": corners,
            }
        )

        return res

    def generate_box_corners(self, img_shape):
        minor_dim = np.min(img_shape[:2])
        if self.idx >= (minor_dim - self.size):
            self.idx = 0
        else:
            self.idx = self.idx + 1

        return np.array(
            (
                (self.idx, self.idx),
                (self.idx, self.idx + self.size),
                (self.idx + self.size, self.idx + self.size),
                (self.idx + self.size, self.idx),
            )
        )
