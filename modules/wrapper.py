import cv2
from PIL import Image
from typing import Any
import io
import numpy as np
from numpy import float_
from numpy._typing import _64Bit
import matplotlib.pyplot as plt

from modules.build_distmap import get_distmap, get_distance
from modules.run_predictions import run_predictions


def wrapper(
        img1: np.ndarray[Any, np.dtype[np.floating[_64Bit] | float_]], img2: np.ndarray[Any, np.dtype[np.floating[_64Bit] | float_]],
        coords: dict[str, np.ndarray[Any, np.dtype[np.floating[_64Bit] | float_]]], models) \
        -> tuple[
            float, np.ndarray[Any, np.dtype[np.floating[_64Bit] | float_]] | np.ndarray[Any, np.dtype[Any]], dict[str, str | bytes]]:

    disp_l, disp_r, volume = run_predictions(img1, img2, models)

    dist_map = get_distmap(cv2.resize(disp_l[0], (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR))
    distance = get_distance(dist_map, coords['red'], coords['blue'])

    threshold = 0.5
    x, y, z = np.where(volume > threshold)  # volume.shape = (32, 32, 32)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.voxels(volume)
    ax.scatter(x, y, z, c=y, cmap='viridis', marker='o')
    ax.set_xlim([0, 32])
    ax.set_ylim([0, 32])
    ax.set_zlim([0, 32])
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    buf = io.BytesIO()
    np.save(buf, volume)
    buf.seek(0)

    return round(distance, 2), image, {"name": "model.npy", "data": buf.getvalue(), "type": "application/octet-stream"}
