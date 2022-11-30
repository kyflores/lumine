import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

CAM0 = '/dev/video3'
CAM1 = '/dev/video5'

BITS=16
def get_cdf(image, number_bins=2**BITS):
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255.0 * cdf / cdf[-1] # normalize
    return cdf, bins

def apply_hist(image, hist_info):
    cdf, bins = hist_info
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    return image_equalized.reshape(image.shape)

def hist_eq(image, number_bins=2**BITS):
    cdf = get_cdf(image, number_bins)
    return apply_hist(image, cdf).astype(np.uint8)


if __name__ == '__main__':
    fl = open('camera_calibrations.pkl', 'rb')
    tmp = pickle.load(fl)
    im0, ipoints0, K0, d0, rvecs0, tvecs0 = tmp[0]
    im1, ipoints1, K1, d1, rvecs1, tvecs1 = tmp[1]

    print(K0)
    print(d0)
    opoints = tmp[2]

    imgsz = (640, 480)

    stcal = cv2.stereoCalibrate(
        opoints,
        ipoints0,
        ipoints1,
        K0,
        d0,
        K1,
        d1,
        imgsz,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    ret, cm0, dc0, cm1, dc1, R, T, E, F = stcal

    strect= cv2.stereoRectify(cm0, dc0, cm1, dc1, imgsz, R, T, flags=cv2.CALIB_ZERO_DISPARITY)
    R1, R2, P1, P2, Q, roi1, roi2 = strect

    (map0_0, map0_1) = cv2.initUndistortRectifyMap(cm0, dc0, R1, P1, imgsz, cv2.CV_32FC1)
    (map1_0, map1_1) = cv2.initUndistortRectifyMap(cm1, dc1, R2, P2, imgsz, cv2.CV_32FC1)

    remap0 = cv2.remap(im0, map0_0, map0_1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    remap1 = cv2.remap(im1, map1_0, map1_1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)

    cam0 = cv2.VideoCapture(CAM0)
    cam0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam1 = cv2.VideoCapture(CAM1)
    cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while(1):
        cam0.grab()
        cam1.grab()
        _, im0 = cam0.retrieve()
        _, im1 = cam1.retrieve()
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        remap0 = cv2.remap(im0, map0_0, map0_1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        remap1 = cv2.remap(im1, map1_0, map1_1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        both = cv2.hconcat((remap0, remap1))
        both = cv2.cvtColor(both, cv2.COLOR_GRAY2BGR)
        sz = both.shape
        both = cv2.line(both, (0, sz[0] // 2), (sz[1], sz[0] // 2), (0,0,255), 1)

        disp_map = stereo.compute(im0, im1)

        disp_map = disp_map.astype(np.float32) / 16
        cv2.imshow('disparity map', disp_map)


        if cv2.pollKey() > -1:
            cam0.release()
            cam1.release()
            cv2.destroyAllWindows()
            break
