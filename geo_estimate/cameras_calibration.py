import os
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibrator
from stereovision.exceptions import ChessboardNotFoundError

rows = 6 # num cols
cols = 9 # num rows
square_size = 2.5*0.3 # size chesssboard squares in cm

img_size = (1920, 1080) # calibration images size

calibrator = StereoCalibrator(rows, cols, square_size, img_size)

images_path = './chessboard/'
img_pairs = os.listdir(images_path)

for pair_path in img_pairs:
    left_name = os.path.join(images_path, pair_path, 'left.png')
    right_name = os.path.join(images_path, pair_path, 'right.png')

    img_left = cv2.imread(left_name, 1)
    img_right = cv2.imread(right_name, 1)

    assert img_left.shape == img_right.shape

    try:
        calibrator._get_corners(img_left)
        calibrator._get_corners(img_right)
    except ChessboardNotFoundError as e:
        print(e)
    else:
        calibrator.add_corners((img_left, img_right), False)


calibration = calibrator.calibrate_cameras()
calibration.export('./cameras_params')

