import cv2
import numpy as np
import detectors.nanodet_common as nc

import os
import tvm.relay as relay
import tvm
from tvm import autotvm, auto_scheduler
from tvm.contrib import graph_executor
import onnx
import time


# https://tvm.apache.org/docs/tutorial/autotvm_relay_x86.html#sphx-glr-tutorial-autotvm-relay-x86-py
# https://github.com/tlc-pack/TLCBench/blob/main/benchmark_autoscheduler.py
class YoloV8TvmDetector:
    def __init__(
        self,
        onnx_file,
        input_tensor_name="images",
        target="llvm",
        tuning_file="tune.json",
        dim=640,
    ):
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
            device_name = "llvm"
        self.device = tvm.device(device_name, 0)
        self.module = graph_executor.GraphModule(self.lib["default"](self.device))

    def detect(self, im):
        im, self.scale = yc.resize_to_frame(im, self.dim)
        blob = cv2.dnn.blobFromImage(
            im,
            1.0 / 255,
            size=(im.shape[1], im.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        print(blob.shape, blob.max(), blob.min())
        self.module.set_input(self.input_tensor_name, blob)
        t_b = time.time()
        self.module.run()

        result_tensor = self.module.get_output(
            0, tvm.nd.empty(self.output_shape)
        ).numpy()
        t_e = time.time()
        print("module.run:", t_e - t_b)

        nms = yc.process_yolov8_output_tensor(result_tensor)
        return yc.boxes_to_detection_dict(nms, self.dim, self.scale)
