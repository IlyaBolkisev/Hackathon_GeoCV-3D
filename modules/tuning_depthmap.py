import os
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibration

# variable_mapping = {
#     "SWS" : 15, 
#     "SpeckleSize" : 100, 
#     "SpeckleRange" : 15, 
#     "UniqRatio" : 10, 
#     "TxtrThrshld" : 100, 
#     "NumofDisp" : 1,
#     "MinDisp": -25, 
#     "PreFiltCap" : 30, 
#     "PreFiltSize" : 105
#     }
# with open('sbm_params.json', 'w') as f:
#     json.dump(variable_mapping, f)

loading = False

def stereo_depth_map(rectified_pair, variable_mapping):

    '''print ('SWS='+str(SWS)+' PFS='+str(PFS)+' PFC='+str(PFC)+' MDS='+\
           str(MDS)+' NOD='+str(NOD)+' TTH='+str(TTH))
    print (' UR='+str(UR)+' SR='+str(SR)+' SPWS='+str(SPWS))'''

    #blockSize is the SAD Window Size
    #Filter settings
    sbm = cv2.StereoBM_create(numDisparities=16, blockSize=variable_mapping["SWS"]) 
    sbm.setPreFilterType(1)    
    sbm.setPreFilterSize(variable_mapping['PreFiltSize'])
    sbm.setPreFilterCap(variable_mapping['PreFiltCap'])
    sbm.setSpeckleRange(variable_mapping['SpeckleRange'])
    sbm.setSpeckleWindowSize(variable_mapping['SpeckleSize'])
    sbm.setMinDisparity(variable_mapping['MinDisp'])
    sbm.setNumDisparities(variable_mapping['NumofDisp'])
    sbm.setTextureThreshold(variable_mapping['TxtrThrshld'])
    sbm.setUniquenessRatio(variable_mapping['UniqRatio'])
    

    c, r = rectified_pair[0].shape
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    #Convering Numpy Array to CV_8UC1
    image = np.array(disparity_normalized, dtype = np.uint8)
    # disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return image, disparity_normalized

def activateTrackbars(x):
    global loading
    loading = False

def create_trackbars() :
    global loading

    #SWS cannot be larger than the image width and image heights.
    #In this case, width = 320 and height = 240
    cv2.createTrackbar("SWS", "Stereo", 5, 230, activateTrackbars)
    cv2.createTrackbar("SpeckleSize", "Stereo", 0, 300, activateTrackbars)
    cv2.createTrackbar("SpeckleRange", "Stereo", 0, 40, activateTrackbars)
    cv2.createTrackbar("UniqRatio", "Stereo", 1, 20, activateTrackbars)
    cv2.createTrackbar("TxtrThrshld", "Stereo", 0, 1000, activateTrackbars)
    cv2.createTrackbar("NumofDisp", "Stereo", 1, 16, activateTrackbars)
    cv2.createTrackbar("MinDisp", "Stereo", -100, 200, activateTrackbars)
    cv2.createTrackbar("PreFiltCap", "Stereo", 1, 63, activateTrackbars)
    cv2.createTrackbar("PreFiltSize", "Stereo", 5, 255, activateTrackbars)
    # cv2.createTrackbar("Save Settings", "Stereo", 0, 1, activateTrackbars)
    # cv2.createTrackbar("Load Settings","Stereo", 0, 1, activateTrackbars)

def onMouse(event, x, y, flag, disparity_normalized):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance = disparity_normalized[y][x]
        print("Distance in centimeters {}".format(distance))



cv2.namedWindow("Stereo", cv2.WINDOW_NORMAL)
create_trackbars()

variables = ["SWS", "SpeckleSize", "SpeckleRange", "UniqRatio", "TxtrThrshld", "NumofDisp",
    "MinDisp", "PreFiltCap", "PreFiltSize"]

variable_mapping = {"SWS" : 15, "SpeckleSize" : 100, "SpeckleRange" : 15, "UniqRatio" : 10, "TxtrThrshld" : 100, "NumofDisp" : 1,
    "MinDisp": -25, "PreFiltCap" : 30, "PreFiltSize" : 105}

left_img = cv2.cvtColor(cv2.imread('./test1001.png'), cv2.COLOR_BGR2GRAY)
right_img = cv2.cvtColor(cv2.imread('./test1002.png'), cv2.COLOR_BGR2GRAY)

while True:
    calibration = StereoCalibration(input_folder='./cameras_params')
    rectified_pair = calibration.rectify((left_img, right_img))

    if loading == False:
        for v in variables:
            current_value = cv2.getTrackbarPos(v, "Stereo")
            if v == "SWS" or v == "PreFiltSize":
                if current_value < 5:
                    current_value = 5
                if current_value % 2 == 0:
                    current_value += 1
            
            if v == "NumofDisp":
                if current_value == 0:
                    current_value = 1
                current_value = current_value * 16
            if v == "MinDisp":
                current_value = current_value - 100
            if v == "UniqRatio" or v == "PreFiltCap":
                if current_value == 0:
                    current_value = 1
            
            variable_mapping[v] = current_value

    current_save = cv2.getTrackbarPos("Save Settings", "Stereo")
    current_load = cv2.getTrackbarPos("Load Settings", "Stereo")

    # save_load_map_settings(current_save, current_load, variable_mapping)
    cv2.setTrackbarPos("Save Settings", "Stereo", 0)
    cv2.setTrackbarPos("Load Settings", "Stereo", 0)
    disparity_color, disparity_normalized = stereo_depth_map(rectified_pair, variable_mapping)

    #What happens when the mouse is clicked
    cv2.setMouseCallback("Stereo", onMouse, disparity_normalized)
    cv2.resizeWindow('Stereo', 1920, 600)
    cv2.imshow('Frame', cv2.resize(disparity_color, (900, 400)))
    # cv2.imshow("Frame", cv2.resize(disparity_color, (900, 450)))
    cv2.imshow("Frame1", cv2.resize(np.hstack((rectified_pair[0], rectified_pair[1])), (1200, 600)))

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    else:
        continue