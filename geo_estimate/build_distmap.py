import cv2
import numpy as np

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
