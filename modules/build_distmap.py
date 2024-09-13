import numpy as np


def get_distmap(disparity, dist_to_obj=641):
    disp = 1 - disparity
    return disp * dist_to_obj + dist_to_obj


def get_distance(distmap, p1, p2, focal_length=80, pixel_size=4.5e-3):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])


    d_obj = (distmap[p1[1], p1[0]] + distmap[p2[1], p2[0]]) / 2 - focal_length

    l_x = dx * d_obj * pixel_size / focal_length
    l_y = dy * d_obj * pixel_size / focal_length
    l_z = abs(distmap[p2[1], p2[0]] - distmap[p1[1], p1[0]]) // 2

    return np.linalg.norm((l_x, l_y, l_z)) * 0.5
