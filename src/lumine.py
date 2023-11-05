import os
import argparse
import time
import concurrent.futures
import json

import numpy as np
import cv2

import subprojects.sort.sort as sort

import common
import tracker as trk
import draw
import camera


class ConfigOpts(object):
    pass


# Initialize detectors here. Everything in this list must implement the detect function
# detailed in the readme and return a list of dictionaries with the right keys.
def get_detectors(opt):
    detectors = []

    from detectors import rpy_apriltags as atg

    detectors.append(
        atg.RobotpyAprilTagDetector(atg.C310_PARAMS, opt.tag_family, opt.tag_size)
    )

    if opt.yolov8_det == 'ultralytics':
        print("Using Ultralytics YoloV8")
        from detectors import yolov8_ultralytics as yolo

        detectors.append(yolo.YoloUltralyticsDetector(opt.weights, dim=opt.yolov8_dim))
    elif opt.yolov8_det == 'openvino':
        print("Using OpenVINO YoloV8")
        from detectors import yolov8_openvino as yolov8_ov

        detectors.append(
            yolov8_ov.YoloV8OpenVinoDetector(opt.weights, backend="AUTO", dim=opt.yolov8_dim)
        )
    elif opt.yolov8_det == 'opencv':
        print("Using OpenCVDNN YoloV8")
        from detectors import yolov8_ocv as yolo_ocv

        detectors.append(yolo_ocv.YoloV8OpenCVDetector(opt.weights, dim=opt.yolov8_dim))
    elif opt.yolov8_det == 'tvm_vk':
        print("Using TVM Vulkan YoloV8")
        from detectors import yolov8_tvm
        if os.path.exists('tune.jsonl'):
            tune = 'tune.jsonl'
        else:
            tune = None

        detectors.append(
            yolov8_tvm.YoloV8TvmDetector(
                opt.weights, target="vulkan", tuning_file=tune, dim=opt.yolov8_dim
            )
        )

    # from detectors import dummy
    # detectors.append(dummy.DummyDetector())

    # from detectors import blob_detector as blob
    # detectors.append(blob.BlobDetector(np.array((30, 150))))

    return detectors


def detect(opt):
    # See https://www.kurokesu.com/main/2020/05/22/uvc-camera-exposure-timing-in-opencv/
    cap = None
    try:
        source = int(opt.source)
        source_type = "webcam"
        print("Configured webcam {} as source".format(source))
    except ValueError:
        source_type = "video"
        source = str(opt.source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Could not open source file")
            exit(1)
        print("Configured file {} as source".format(source))
    except:
        print("Unknown error parsing source")
        exit(1)

    # camera_res = (720, 960)

    # 1280x960 max res for C310
    camera_res = (960, 1280)

    if source_type == "webcam":
        cap = camera.CameraCtl(source, camera_res, 30, opt.gain, opt.exposure)

    lumine_table = None
    if opt.nt:
        # Don't import if the option is false to avoid needing this dep for development.
        import ntcore_formatter

        lumine_table = ntcore_formatter.NtcoreFormatter(opt.nt)

    stream = None
    if opt.stream:
        print(opt.stream)
        import cscore_stream

        stream = cscore_stream.CsCoreStream((240, 320), opt.iface, opt.stream, fps=15)

    detectors = get_detectors(opt)

    tracker = trk.Sort(opt.max_age, opt.min_hits, opt.iou_thresh)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            err, frame = cap.read()
            if not err:
                print("Media source didn't produce frame, stopping...")
                break

            t_begin = time.time()
            # Note: This uses a threadpool executor to start each detection
            # concurrently. Despite GIL, this should still improve performance
            # because it will at least allow us to do other work like apriltags
            # while we wait for a DNN accelerator to produce a result
            asyncres = [executor.submit(x.detect, frame) for x in detectors]

            all_dets = []
            for res in asyncres:
                all_dets = all_dets + res.result()

            all_dets = [x for x in all_dets if (x["confidence"] >= opt.conf_thresh)]
            all_dets.sort(key=lambda x: x["confidence"], reverse=True)

            all_dets = tracker.update(all_dets)

            with_boxes = draw.draw(frame, all_dets)
            # with_boxes = draw.draw_sort(frame, trackers)

            t_end = time.time()

            if opt.draw:
                cv2.imshow("detector", with_boxes)

            if opt.stream:
                stream.write_frame(with_boxes)

            # Update network tables if it was enabled.
            if lumine_table is not None:
                y_res = executor.submit(lumine_table.update_yolo, all_dets)
                a_res = executor.submit(lumine_table.update_apriltags_rpy, all_dets)
                a_res.result()
                y_res.result()

            # Print out the formatted ASCII table if it was requested.
            if opt.table:
                os.system("cls" if os.name == "nt" else "clear")
                ms = 1000 * (t_end - t_begin)
                fps = 1 / (t_end - t_begin)
                print("Took {:.2f} ms, {:.2f} iter/sec".format(ms, fps))
                print(common.detections_as_table(all_dets))

            if cv2.pollKey() > -1:
                cv2.destroyAllWindows()
                break


def main():
    parser = argparse.ArgumentParser()

    # Import arguments for basic operation.
    parser.add_argument(
        "--config",
        type=str,
        help="Config file path. If config is passed, all CLI arguments are ignored! Config file is a json with keys that are the same as CLI argument names.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=0,
        help="Media source, anything supported by OpenCV's video capture",
    )
    parser.add_argument(
        "--weights", type=str, default="yolov5s.pt", help="Path to YOLO weights file"
    )
    parser.add_argument(
        "--table", action="store_true", help="Print the detection table."
    )
    parser.add_argument(
        "--draw", action="store_true", help="Draw the augmented feed to a window."
    )
    parser.add_argument(
        "--nt",
        type=str,
        help="IP of the network tables server. If it's a team number, will be automatically set to 10.TE.AM.2, else will be parsed as a full IP.",
    )
    parser.add_argument(
        "--iface",
        type=str,
        default="eth0",
        help="Name of the interface connected to the robot. It will be something like eth0, en0, enps0. Lumine will use the first IPv4 addr on this interface.",
    )
    parser.add_argument(
        "--stream",
        type=int,
        help="Port to start cscore on.",
    )

    # Secondary arguments concered with camera tuning.
    parser.add_argument(
        "--gain", type=int, default=150, help="Gain to configure with v4l2-ctl"
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=500,
        help="Exposure time to configure with v4l2-ctl",
    )
    parser.add_argument(
        "--tag_family", type=str, default="tag16h5", help="Apriltag family"
    )
    parser.add_argument(
        "--tag_size", type=float, default=1.0, help="Apriltag size in meters"
    )
    parser.add_argument(
        "--max_age",
        type=int,
        default=30,
        help="Longest time (in frames) SORT will remember an ID without a detection",
    )
    parser.add_argument(
        "--min_hits",
        type=int,
        default=1,
        help="Minimum number of consecutive needed to start tracking something",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.15,
        help="IOU threshold for SORT. Smaller can track faster movements but reduces accuracy",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.50,
        help="Confidence threshold for processing a detect.",
    )
    parser.add_argument(
        "--yolov_dim",
        type=int,
        default=640,
        help="imgsz of the yolov8 model",
    )
    parser.add_argument(
        "--yolov8_det",
        type=str,
        choices=['ultralytics, openvino, opencv, tvm_vk'],
        default='ultralytics',
        help="Backend to use for running yolov8 models."
    )

    opt = parser.parse_args()

    if opt.config is not None:
        with open(opt.config, "r") as f:
            config = json.load(f)

        opt = ConfigOpts()
        for k in config.keys():
            setattr(opt, k, config[k])

    detect(opt)
    print("Detector exited.")
    exit(0)


if __name__ == "__main__":
    main()
