import base64

from fastapi import FastAPI, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import Dict, Annotated
import numpy as np
from PIL import Image
import io
import logging

from modules.run_predictions import warmup_models
from modules.wrapper import wrapper

models = warmup_models('./weights')

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
        main_img = Image.open(io.BytesIO(await image1.read())).convert('RGB')
        img2 = Image.open(io.BytesIO(await image2.read())).convert('RGB')
        coords1 = Coordinates.parse_raw(coords1)
        # Convert the image to numpy arrays
        main_img_np = np.array(main_img)
        img2_np = np.array(img2)

        # Convert coordinates to numpy arrays
        coords_np = {
            "red": np.array([coords1.red["x"], coords1.red["y"]], dtype=int),
            "blue": np.array([coords1.blue["x"], coords1.blue["y"]], dtype=int)
        }

        # Calculate distances
        distance, screenshot_np, model = wrapper(main_img_np, img2_np, coords_np, models)
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
