import os
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibration

calibration = StereoCalibration(input_folder='./cameras_params')

img_left = cv2.cvtColor(cv2.imread('./left/rotateX_0001002.png', 1), cv2.COLOR_BGR2RGB)
img_right = cv2.cvtColor(cv2.imread('./right/rotateX_0001001.png', 1), cv2.COLOR_BGR2RGB)

rectified_pair = calibration.rectify((img_left, img_right))

cv2.imwrite('./rectified_left.jpg', rectified_pair[0])
cv2.imwrite('./rectified_right.jpg', rectified_pair[1])
