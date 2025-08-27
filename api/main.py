import io
from contextlib import asynccontextmanager

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_HEIGHT = 256
IMG_WIDTH = 256
COLOR_MAP = [
    [0, 0, 0],
    [0, 255, 255],
    [255, 0, 0],
    [153, 76, 0],
    [0, 153, 0],
]

scaled_color_map = [[c[0] / 255.0, c[1] / 255.0, c[2] / 255.0]
                    for c in COLOR_MAP]
cmap = mcolors.ListedColormap(scaled_color_map)

unet_model = None
deeplab_model = None


def load_models():
    global unet_model, deeplab_model
    try:
        unet_model = tf.keras.models.load_model("unet_model.h5", compile=False)
        deeplab_model = tf.keras.models.load_model(
            "deeplab_model.h5", compile=False)
    except Exception as e:
        print(f"Error loading models: {e}")


@app.on_event("startup")
async def startup_event():
    load_models()


@app.get("/")
async def root():
    return {"message": "Oil Spill Segmentation API"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": unet_model is not None and deeplab_model is not None,
    }


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = image / 255.0
    return image


def create_prediction_plot(original_image, predicted_mask, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(
        predicted_mask, cmap=cmap, vmin=0, vmax=len(COLOR_MAP) - 1, interpolation="none"
    )
    axes[1].set_title(f"Predicted Mask ({model_name})")
    axes[1].axis("off")

    axes[2].imshow(original_image)
    axes[2].imshow(
        predicted_mask,
        cmap=cmap,
        alpha=0.5,
        vmin=0,
        vmax=len(COLOR_MAP) - 1,
        interpolation="none",
    )
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close()

    return buf


@app.post("/predict/unet")
async def predict_unet(file: UploadFile = File(...)):
    if not unet_model:
        raise HTTPException(status_code=500, detail="UNet model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)

        prediction = unet_model.predict(
            np.expand_dims(processed_image, axis=0))
        predicted_mask = np.argmax(prediction, axis=3)[0, :, :]

        plot_buffer = create_prediction_plot(
            processed_image, predicted_mask, "UNet")

        return StreamingResponse(
            io.BytesIO(plot_buffer.read()),
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=unet_prediction.png"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/deeplab")
async def predict_deeplab(file: UploadFile = File(...)):
    if not deeplab_model:
        raise HTTPException(status_code=500, detail="DeepLab model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)

        prediction = deeplab_model.predict(
            np.expand_dims(processed_image, axis=0))
        predicted_mask = np.argmax(prediction, axis=3)[0, :, :]

        plot_buffer = create_prediction_plot(
            processed_image, predicted_mask, "DeepLabV3+"
        )

        return StreamingResponse(
            io.BytesIO(plot_buffer.read()),
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=deeplab_prediction.png"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/both")
async def predict_both(file: UploadFile = File(...)):
    if not unet_model or not deeplab_model:
        raise HTTPException(status_code=500, detail="Models not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)

        unet_prediction = unet_model.predict(
            np.expand_dims(processed_image, axis=0))
        unet_mask = np.argmax(unet_prediction, axis=3)[0, :, :]

        deeplab_prediction = deeplab_model.predict(
            np.expand_dims(processed_image, axis=0)
        )
        deeplab_mask = np.argmax(deeplab_prediction, axis=3)[0, :, :]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(processed_image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(
            unet_mask, cmap=cmap, vmin=0, vmax=len(COLOR_MAP) - 1, interpolation="none"
        )
        axes[0, 1].set_title("UNet Prediction")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(processed_image)
        axes[0, 2].imshow(
            unet_mask,
            cmap=cmap,
            alpha=0.5,
            vmin=0,
            vmax=len(COLOR_MAP) - 1,
            interpolation="none",
        )
        axes[0, 2].set_title("UNet Overlay")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(processed_image)
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(
            deeplab_mask,
            cmap=cmap,
            vmin=0,
            vmax=len(COLOR_MAP) - 1,
            interpolation="none",
        )
        axes[1, 1].set_title("DeepLabV3+ Prediction")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(processed_image)
        axes[1, 2].imshow(
            deeplab_mask,
            cmap=cmap,
            alpha=0.5,
            vmin=0,
            vmax=len(COLOR_MAP) - 1,
            interpolation="none",
        )
        axes[1, 2].set_title("DeepLabV3+ Overlay")
        axes[1, 2].axis("off")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        return StreamingResponse(
            io.BytesIO(buf.read()),
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=both_predictions.png"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

