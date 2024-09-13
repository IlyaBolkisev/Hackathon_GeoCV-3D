import cv2
import numpy as np

left_img = cv2.imread('geo_estimate/left1/cam1_1.jpg', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('geo_estimate/right1/Cam2_1.jpg', cv2.IMREAD_GRAYSCALE)

# Настройка стерео-блока для расчета карты диспаратности
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity_map = stereo.compute(left_img, right_img)

disp_diff = disparity_map.max() - disparity_map.min()

disparity_map = disparity_map.astype('float') / disp_diff.astype('float')

dist_to_obj = 641 # mm

dist_map = disparity_map * dist_to_obj + dist_to_obj

print(dist_map.min(), dist_map.max())

def get_distmap(disparity, dist_to_obj):
    disp = 1 - disparity
    return disp*dist_to_obj + dist_to_obj

def get_distance(distmap, p1, p2, focal_length=80, pixel_size=4500):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    center_x = (p1[0] + p2[0]) / 2
    center_y = (p1[1] + p2[1]) / 2
    
    d_obj = distmap[center_y, center_x]

    l_x = dx * d_obj * pixel_size / (focal_length * distmap.shape[1])
    l_y = dy * d_obj * pixel_size / (focal_length * distmap.shape[0])
    l_z = distmap[p2[1], p2[0]] - distmap[p1[1], p1[0]]

    return np.linalg.norm((l_x, l_y, l_z))
