import pickle

import cv2
import numpy as np

CAM0 = '/dev/video3'
CAM1 = '/dev/video5'
BOARD_SHAPE=(8,6)
SQUARE_SIZE=24 # mm
NUM_FRAMES=10

CAL_FLAGS = (cv2.CALIB_ZERO_TANGENT_DIST |
         cv2.CALIB_FIX_K1 |
         cv2.CALIB_FIX_K2 |
         cv2.CALIB_FIX_K3 |
         cv2.CALIB_FIX_K4 |
         cv2.CALIB_FIX_K5 |
         cv2.CALIB_FIX_K6)

def chessboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, corners = cv2.findChessboardCorners(img, BOARD_SHAPE)
    overlay = cv2.drawChessboardCorners(img, BOARD_SHAPE, corners, retval)
    return corners, overlay

if __name__ == '__main__':
    cam0 = cv2.VideoCapture(CAM0)
    cam1 = cv2.VideoCapture(CAM1)
    cam0.open(CAM0)
    cam0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam1.open(CAM1)
    cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    x = np.arange(BOARD_SHAPE[0]) * SQUARE_SIZE
    y = np.arange(BOARD_SHAPE[1]) * SQUARE_SIZE

    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.zeros_like(xgrid)
    opoints = np.dstack((xgrid, ygrid, zgrid)).reshape((-1, 1, 3)).astype(np.float32)

    ipoints0 = []
    ipoints1 = []
    imgsz = (480, 640)
    im0 = None
    im1 = None
    for ix in range(NUM_FRAMES):

        # Reopen the camera on each loop iteration. Opening it once
        # and trying to repeatedly read() seems to pull from a buffer of images,
        # the image it returns was clearly not captured at that moment.

        while(1):
            cam0.grab()
            cam1.grab()
            _, im0 = cam0.retrieve()
            _, im1 = cam1.retrieve()
            cv2.imshow('cal', cv2.hconcat((im0, im1)))
            if cv2.pollKey() > -1:
                print("Captured {}".format(ix))
                break

        assert im0.shape == im1.shape

        corner0, ov0 = chessboard(im0)
        corner1, ov1 = chessboard(im1)

        ipoints0.append(corner0)
        ipoints1.append(corner1)

    assert(len(ipoints0) == len(ipoints1))


    opoints = [opoints] * len(ipoints0)
    ret, K0, d0, rvecs0, tvecs0 = cv2.calibrateCamera(
            opoints,
            ipoints0,
            imgsz,
            cameraMatrix=None,
            distCoeffs=np.zeros(5),
            flags=CAL_FLAGS)

    ret, K1, d1, rvecs1, tvecs1 = cv2.calibrateCamera(
            opoints,
            ipoints1,
            imgsz,
            cameraMatrix=None,
            distCoeffs=np.zeros(5),
            flags=CAL_FLAGS)

    print('K0')
    print(K0)
    print('K1')
    print(K1)

    fl = open('camera_calibrations.pkl', 'wb')
    pickle.dump([
        (im0, ipoints0, K0, d0, rvecs0, tvecs0),
        (im1, ipoints1, K1, d1, rvecs1, tvecs1),
        opoints],fl)

