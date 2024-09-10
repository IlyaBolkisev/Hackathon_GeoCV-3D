from fastapi import FastAPI, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import Dict, Annotated
import numpy as np
# from geo_estimate import calculate_3d_distance
from PIL import Image
import io
import logging
import uvicorn


# TODO replace with real func
def wrapper(img1, img2, coords1, coords2):
    return 1.1, np.array([0])


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
        distance = wrapper(img1_np, img2_np, coords1_np, coords2_np)

        return {"distance": distance}

    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
