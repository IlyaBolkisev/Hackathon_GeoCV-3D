import base64
import time

from fastapi import FastAPI, UploadFile, HTTPException, Form
from numpy import ndarray, dtype, floating, float_, unsignedinteger
from numpy._typing import _64Bit, _8Bit
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
        coords: dict[str, ndarray[Any, dtype[floating[_64Bit] | float_]]]) \
        -> tuple[
            float, ndarray[Any, dtype[floating[_64Bit] | float_]] | ndarray[Any, dtype[Any]], dict[str, str | bytes]]:
    w, h = 512, 512
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data[0:256, 0:256] = [255, 0, 0]  # red patch in upper left
    img = Image.fromarray(data, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 123.45, data, {"name": "model.png", "data": buf.getvalue(), "type": "image/png"}


app = FastAPI()


class Coordinates(BaseModel):
    red: Dict[str, float]
    blue: Dict[str, float]


@app.post("/api/v1/calculate-distance/")
async def calculate_distance(
    image1: UploadFile,
    image2: UploadFile,
    coords1: Annotated[str, Form()],
):
    try:
        # Convert uploaded images to PIL Image
        main_img = Image.open(io.BytesIO(await image1.read()))
        img2 = Image.open(io.BytesIO(await image2.read()))
        coords1 = Coordinates.parse_raw(coords1)
        # Convert the image to numpy arrays
        main_img_np = np.array(main_img)
        img2_np = np.array(img2)

        # Convert coordinates to numpy arrays
        coords_np = {
            "red": np.array([coords1.red["x"], coords1.red["y"]]),
            "blue": np.array([coords1.blue["x"], coords1.blue["y"]])
        }

        # Calculate distances
        distance, screenshot_np, model = calculate_3d_distance(main_img_np, img2_np, coords_np)
        screenshot = Image.fromarray(screenshot_np)
        screenshot_buf = io.BytesIO()
        screenshot.save(screenshot_buf, format='PNG')

        return {
            "distance": distance,
            "screenshot": {
                "name": "screenshot.png",
                "data": base64.b64encode(screenshot_buf.getvalue()).decode("utf-8"),
                "type": "image/png"
            },
            "model": {
                "name": model["name"],
                "data": base64.b64encode(model["data"]).decode("utf-8"),
                "type": model["type"]
            }}

    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500)
