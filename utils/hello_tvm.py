# Get an onnx model. For instance:
# `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
# `pip install ultralytics`
# `pip install onnx xgboost==1.5.2` # Needed by tvm tune
# `yolo export model=yolov8n.pt format=onnx`
# produces yolov8n.onnx
# https://tvm.apache.org/docs/tutorial/tvmc_python.html#sphx-glr-tutorial-tvmc-python-py
# export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/yourloader.json

from tvm.driver import tvmc

model = tvmc.load("yolov8n.onnx")

tune = "tune.json"
target = "vulkan"
# target = "vulkan -supports_float16=true"

print("Running model first with no tuning...")
package = tvmc.compile(model, target=target, tuning_records=tune)
result = tvmc.run(package, device=target, benchmark=True, repeat=20)
print(result.format_times())

print("Tuning Model...")
tvmc.tune(
    model,
    target=target,
    tuning_records=tune,
    # prior_records=tune,
    enable_autoscheduler=True,
    trials=3000)
package = tvmc.compile(model, target=target, tuning_records=tune)
print("Running model with tuning ...")
result = tvmc.run(package, device=target, benchmark=True, repeat=20)
print(result.format_times())
