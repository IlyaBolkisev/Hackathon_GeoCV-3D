import time

from fastapi import FastAPI, UploadFile, HTTPException, Form
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from pydantic import BaseModel
from typing import Dict, Annotated, Tuple, Any
import numpy as np
# from geo_estimate import calculate_3d_distance
from PIL import Image
import io
import logging


# TODO replace with real func
# ndarray sizes are made up. don't know the real sizes
def calculate_3d_distance(
        img1: ndarray[Any, dtype[floating[_64Bit] | float_]], img2: ndarray[Any, dtype[floating[_64Bit] | float_]],
        coords1: dict[str, ndarray[Any, dtype[floating[_64Bit] | float_]]],
        coords2: dict[str, ndarray[Any, dtype[floating[_64Bit] | float_]]]) \
        -> tuple[float, ndarray[Any, dtype[floating[_64Bit] | float_]]]:
    mesh = np.zeros([50, 50, 50])
    for i in range(len(mesh)):
        for j in range(len(mesh[i])):
            for k in range(len(mesh[i][j])):
                if i == 0 or i == 49 or j == 0 or j == 49 or k == 0 or k == 49:
                    mesh[i][j][k] = 255
    time.sleep(1)
    return 123.45, mesh


app = FastAPI()


class Coordinates(BaseModel):
    red: Dict[str, float]
    blue: Dict[str, float]


@app.post("/api/v1/calculate-distance/")
async def calculate_distance(
    image1: UploadFile,
    image2: UploadFile,
    coords1: Annotated[str, Form()],
    coords2: Annotated[str, Form()],
):
    try:
        # Convert uploaded images to PIL Image
        img1 = Image.open(io.BytesIO(await image1.read()))
        img2 = Image.open(io.BytesIO(await image2.read()))
        coords1 = Coordinates.parse_raw(coords1)
        coords2 = Coordinates.parse_raw(coords2)
        # Convert the image to numpy arrays
        img1_np = np.array(img1)
        img2_np = np.array(img2)

        # Convert coordinates to numpy arrays
        coords1_np = {
            "red": np.array([coords1.red["x"], coords1.red["y"]]),
            "blue": np.array([coords1.blue["x"], coords1.blue["y"]])
        }
        coords2_np = {
            "red": np.array([coords2.red["x"], coords2.red["y"]]),
            "blue": np.array([coords2.blue["x"], coords2.blue["y"]])
        }

        # Calculate distances
        distance, mesh = calculate_3d_distance(img1_np, img2_np, coords1_np, coords2_np)
        mesh = mesh.tolist()
        voxels = []
        for i in range(len(mesh)):
            for j in range(len(mesh[i])):
                for k in range(len(mesh[i][j])):
                    if mesh[i][j][k] != 0:
                        voxels.append([i, j, k])

        return {"distance": distance, "mesh": voxels}

    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500)
