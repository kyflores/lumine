from detectors import yolov8_tvm as yolov8_tvm
import numpy as np
import cv2

# img = cv2.imread("")
test_data = np.random.randint(0, 255, (1, 3, 640, 640), dtype=np.uint8)
det = yolov8_tvm.YoloV8TvmDetector("yolov8n.onnx", target="vulkan", dim=640)

res = det.detect(img)
print(res)
