import os
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibration
from matplotlib import pyplot as plt

def stereo_depth_map(img_pair, sbm):
    img_left = img_pair[0]
    img_right = img_pair[1]

    disparity = sbm.compute(img_left, img_right)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    image = np.array(disparity_normalized, dtype=np.uint8)
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    return disparity_color, disparity_normalized

left_img = cv2.cvtColor(cv2.imread('./chessboard/1/left.png'), cv2.COLOR_BGR2GRAY)
right_img = cv2.cvtColor(cv2.imread('./chessboard/1/right.png'), cv2.COLOR_BGR2GRAY)

calibration = StereoCalibration(input_folder='./cameras_params')
rectified_pair = calibration.rectify((left_img, right_img))

with open('sbm_params.json', 'r') as f:
    sbm_params = json.load(f)

sbm = cv2.StereoBM_create(numDisparities=16, blockSize=sbm_params['SWS'])
sbm.setPreFilterType(1)
sbm.setPreFilterSize(sbm_params['PreFiltSize'])
sbm.setPreFilterCap(sbm_params['PreFiltCap'])
sbm.setMinDisparity(sbm_params['MinDisp'])
sbm.setNumDisparities(sbm_params['NumofDisp'])
sbm.setTextureThreshold(sbm_params['TxtrThrshld'])
sbm.setUniquenessRatio(sbm_params['UniqRatio'])
sbm.setSpeckleRange(sbm_params['SpeckleRange'])
sbm.setSpeckleWindowSize(sbm_params['SpeckleSize'])

disparity_color, disparity_normalized = stereo_depth_map(rectified_pair, sbm)

cv2.imwrite('./disparity.jpg', cv2.cvtColor(disparity_normalized, cv2.COLOR_BGR2GRAY))