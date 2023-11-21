import cv2
import numpy as np
import os
import detectors.nanodet_common as nc
import detectors.yolo_common as yc

import tvm.relay as relay
import tvm
from tvm import autotvm, auto_scheduler
from tvm.contrib import graph_executor
import onnx
import time


# nanodet.onnx from nanodet-plus-m_416
# Input: data (1, 3, 416, 416)
# Output output (1, 3598, 112) for COCO 80 classes
class NanodetTvmDetector:
    def __init__(self, onnx_file, input_tensor_name="data", target="llvm", tuning_file="tune.jsonl", dim=416):
        self.input_tensor_name = input_tensor_name
        self.shape_dict = {self.input_tensor_name: (1, 3, dim, dim)}
        self.onnx_model = onnx.load(onnx_file)
        self.output_shape = [
            d.dim_value
            for d in self.onnx_model.graph.output[0].type.tensor_type.shape.dim
        ]

        self.scale = 1.0
        self.dim = dim
        self.target = target

        self.mod, self.params = relay.frontend.from_onnx(
            self.onnx_model, self.shape_dict
        )

        print("Compiling model...")
        if tuning_file is not None:
            # with autotvm.apply_history_best(tuning_file):
            with auto_scheduler.ApplyHistoryBest(
                tuning_file
            ):  # This is a different autotuning mode!
                with tvm.transform.PassContext(
                    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
                ):
                    self.lib = relay.build(
                        self.mod, target=self.target, params=self.params
                    )
        else:
            with tvm.transform.PassContext(opt_level=3):
                self.lib = relay.build(self.mod, target=self.target, params=self.params)

        if self.target.startswith("vulkan"):
            device_name = "vulkan"
        elif self.target.startswith("opencl"):
            device_name = "cl"
        elif self.target.startswith("llvm"):
            device_name = "cpu"
        self.device = tvm.device(device_name, 0)
        self.module = graph_executor.GraphModule(self.lib["default"](self.device))

        self.grid = nc.generate_grid_center_priors(dim, dim)

    def detect(self, im):
        im, self.scale = yc.resize_to_frame(im, self.dim)
        blob = cv2.dnn.blobFromImage(
            im,
            1.0 / 57.63,
            size=(im.shape[1], im.shape[0]),
            mean=(103.53, 116.28, 123.675),
            swapRB=True,
            crop=False,
        )

        self.module.set_input(self.input_tensor_name, blob)
        t_b = time.time()
        self.module.run()

        result_tensor = self.module.get_output(
            0, tvm.nd.empty(self.output_shape)
        ).numpy()
        t_e = time.time()
        print("module.run:", t_e - t_b)

        nms = nc.decode_infer(self.dim, self.dim, result_tensor, self.grid, 0.25)

        return yc.boxes_to_detection_dict(nms, self.dim, self.scale)
