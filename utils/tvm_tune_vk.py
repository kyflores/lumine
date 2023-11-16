# Get an onnx model. For instance:
# `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
# `pip install ultralytics`
# `pip install onnx xgboost==1.5.2` # Needed by tvm tune
# `yolo export model=yolov8n.pt format=onnx`
# produces yolov8n.onnx
# https://tvm.apache.org/docs/tutorial/tvmc_python.html#sphx-glr-tutorial-tvmc-python-py
# export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/yourloader.json

import argparse
from tvm.driver import tvmc


def tune(opt):
    model_nm = opt.model
    tune = opt.tune_file
    target = "vulkan -from_device=0"

    model = tvmc.load(model_nm)
    print("Running model first with no tuning")
    package = tvmc.compile(model, target=target, tuning_records=None)
    result = tvmc.run(package, device="vulkan", benchmark=True, repeat=50)
    print(result.format_times())

    print("Tuning Model")
    tvmc.tune(
        model,
        target=target,
        tuning_records=tune,
        enable_autoscheduler=(opt.tune_type == "autoscheduler"),
        trials=opt.trials,
    )


def test(opt):
    model_nm = opt.model
    tune = opt.tune_file
    target = "vulkan -from_device=0"
    model = tvmc.load(model_nm)

    print("Running model first with no tuning")
    untuned_package = tvmc.compile(model, target=target, tuning_records=None)
    untuned_result = tvmc.run(
        untuned_package, device="vulkan", benchmark=True, repeat=50
    )
    print(untuned_result.format_times())

    context = (
        ["relay.backend.use_auto_scheduler=1"]
        if (opt.tune_type == "autoscheduler")
        else []
    )
    print("Compiling tuned model")
    package = tvmc.compile(
        model, target=target, tuning_records=tune, pass_context_configs=context
    )
    print("Running tuned model")
    result = tvmc.run(package, device="vulkan", benchmark=True, repeat=50)
    print(result.format_times())


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="yolov8n.onnx",
    help="Model name.",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["tune", "test"],
    default="tune",
    help="tune to tune a model, test to evaluate its performance.",
)
parser.add_argument(
    "--tune_type",
    type=str,
    choices=["autotvm", "autoscheduler"],
    default="autoscheduler",
    help="Tuning type to use.",
)
parser.add_argument(
    "--tune_file",
    type=str,
    default="tune.jsonl",
    help="Log file name of the tuning results.",
)
parser.add_argument(
    "--trials",
    type=int,
    default=3000,
    help="Number of trials to run when searching for the best tuning.",
)

opt = parser.parse_args()

if opt.mode == "tune":
    tune(opt)
elif opt.mode == "test":
    test(opt)
