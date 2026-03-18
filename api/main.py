import io
import os
import tempfile
from contextlib import asynccontextmanager

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageOps


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
    # Expose custom detection headers so the browser can read them
    expose_headers=[
        "X-Oil-Spill-Present",
        "X-Oil-Spill-Fraction",
        "X-Oil-Spill-Pixels",
        "X-Oil-Spill-Total-Pixels",
        "X-Oil-Spill-Area-Km2",
        "X-Oil-Spill-Present-UNet",
        "X-Oil-Spill-Fraction-UNet",
        "X-Oil-Spill-Pixels-UNet",
        "X-Oil-Spill-Total-Pixels-UNet",
        "X-Oil-Spill-Area-Km2-UNet",
        "X-Oil-Spill-Present-DeepLab",
        "X-Oil-Spill-Fraction-DeepLab",
        "X-Oil-Spill-Pixels-DeepLab",
        "X-Oil-Spill-Total-Pixels-DeepLab",
        "X-Oil-Spill-Area-Km2-DeepLab",
    ],
)

IMG_HEIGHT = 256
IMG_WIDTH = 256
# Color mapping for segmentation classes (RGB values)
COLOR_MAP = [
    [0, 0, 0],      # Class 0: Background - Black
    [0, 255, 255],  # Class 1: Sheen - Cyan
    [255, 0, 0],    # Class 2: Oil Spill - Red
    [153, 76, 0],   # Class 3: Ship - Brown
    [0, 153, 0],    # Class 4: Land/Vegetation - Green
]

scaled_color_map = [[c[0] / 255.0, c[1] / 255.0, c[2] / 255.0] for c in COLOR_MAP]
cmap = mcolors.ListedColormap(scaled_color_map)

unet_model = None
deeplab_model = None
sarvsdrone_model = None


def load_models():
    global unet_model, deeplab_model
    try:
        unet_model = tf.keras.models.load_model("unet_model.h5", compile=False)
        deeplab_model = tf.keras.models.load_model("deeplab_model.h5", compile=False)
        # Load SAR vs Drone classifier if available
        try:
            sarvsdrone_model = tf.keras.models.load_model(
                "SARVersusDroneModel.h5", compile=False
            )
            globals()["sarvsdrone_model"] = sarvsdrone_model
        except Exception:
            print("SAR vs Drone model not found or failed to load")
    except Exception as e:
        print(f"Error loading models: {e}")

    # Optionally preload any external clients if needed in future


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


def compute_oil_spill_stats(
    predicted_mask: np.ndarray, pollution_classes: tuple[int, ...] = (1, 2)
):
    """
    Compute simple oil / sheen pollution statistics from a predicted mask.

    By default we treat both:
      - Class 1 (Sheen)
      - Class 2 (Oil Spill)
    as positive "oil-related" detections.

    We treat ANY pixel in one of the pollution classes as a positive
    detection. In other words, the minimum fraction threshold is 0.0:
    even the tiniest contiguous region will be reported as present.

    Area estimation:
      - Uses a ground sampling distance (GSD) in meters per pixel side.
      - Configure via env var OIL_SPILL_PIXEL_SIZE_METERS (default 10.0).
      - Area per pixel = GSD^2, converted from m² to km².

    Returns a tuple:
        (
            has_spill: bool,
            fraction: float,
            pollution_pixels: int,
            total_pixels: int,
            area_km2: float,
        )
    """
    try:
        if not isinstance(pollution_classes, (list, tuple, set)):
            pollution_classes = (int(pollution_classes),)

        mask = np.isin(predicted_mask, list(pollution_classes))
        pollution_pixels = int(np.sum(mask))
        total_pixels = int(predicted_mask.size)
        fraction = float(pollution_pixels) / float(total_pixels) if total_pixels else 0.0

        # Any detected pollution counts as positive
        has_spill = pollution_pixels > 0

        # Approximate physical area in km² based on pixel size metadata
        try:
            pixel_size_m = float(os.getenv("OIL_SPILL_PIXEL_SIZE_METERS", "10.0"))
        except Exception:
            pixel_size_m = 10.0

        if pixel_size_m < 0:
            pixel_size_m = 0.0

        area_per_pixel_m2 = pixel_size_m * pixel_size_m
        area_km2 = (pollution_pixels * area_per_pixel_m2) / 1_000_000.0

        return has_spill, fraction, pollution_pixels, total_pixels, area_km2
    except Exception:
        # On any unexpected error, fall back to no spill
        total = int(predicted_mask.size or 0)
        return False, 0.0, 0, total, 0.0


@app.post("/predict/unet")
async def predict_unet(file: UploadFile = File(...)):
    if not unet_model:
        raise HTTPException(status_code=500, detail="UNet model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)

        prediction = unet_model.predict(np.expand_dims(processed_image, axis=0))
        predicted_mask = np.argmax(prediction, axis=3)[0, :, :]

        has_spill, fraction, oil_pixels, total_pixels, area_km2 = (
            compute_oil_spill_stats(predicted_mask)
        )

        plot_buffer = create_prediction_plot(processed_image, predicted_mask, "UNet")

        headers = {
            "Content-Disposition": "attachment; filename=unet_prediction.png",
            "X-Oil-Spill-Present": "true" if has_spill else "false",
            "X-Oil-Spill-Fraction": str(fraction),
            "X-Oil-Spill-Pixels": str(oil_pixels),
            "X-Oil-Spill-Total-Pixels": str(total_pixels),
            "X-Oil-Spill-Area-Km2": str(area_km2),
        }

        return StreamingResponse(
            io.BytesIO(plot_buffer.read()),
            media_type="image/png",
            headers=headers,
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

        prediction = deeplab_model.predict(np.expand_dims(processed_image, axis=0))
        predicted_mask = np.argmax(prediction, axis=3)[0, :, :]

        has_spill, fraction, oil_pixels, total_pixels, area_km2 = (
            compute_oil_spill_stats(predicted_mask)
        )

        plot_buffer = create_prediction_plot(
            processed_image, predicted_mask, "DeepLabV3+"
        )

        headers = {
            "Content-Disposition": "attachment; filename=deeplab_prediction.png",
            "X-Oil-Spill-Present": "true" if has_spill else "false",
            "X-Oil-Spill-Fraction": str(fraction),
            "X-Oil-Spill-Pixels": str(oil_pixels),
            "X-Oil-Spill-Total-Pixels": str(total_pixels),
            "X-Oil-Spill-Area-Km2": str(area_km2),
        }

        return StreamingResponse(
            io.BytesIO(plot_buffer.read()),
            media_type="image/png",
            headers=headers,
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

        unet_prediction = unet_model.predict(np.expand_dims(processed_image, axis=0))
        unet_mask = np.argmax(unet_prediction, axis=3)[0, :, :]

        deeplab_prediction = deeplab_model.predict(
            np.expand_dims(processed_image, axis=0)
        )
        deeplab_mask = np.argmax(deeplab_prediction, axis=3)[0, :, :]

        (
            unet_has_spill,
            unet_fraction,
            unet_oil_pixels,
            unet_total_pixels,
            unet_area_km2,
        ) = compute_oil_spill_stats(unet_mask)
        (
            deeplab_has_spill,
            deeplab_fraction,
            deeplab_oil_pixels,
            deeplab_total_pixels,
            deeplab_area_km2,
        ) = compute_oil_spill_stats(deeplab_mask)

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

        headers = {
            "Content-Disposition": "attachment; filename=both_predictions.png",
            # UNet stats
            "X-Oil-Spill-Present-UNet": "true" if unet_has_spill else "false",
            "X-Oil-Spill-Fraction-UNet": str(unet_fraction),
            "X-Oil-Spill-Pixels-UNet": str(unet_oil_pixels),
            "X-Oil-Spill-Total-Pixels-UNet": str(unet_total_pixels),
            "X-Oil-Spill-Area-Km2-UNet": str(unet_area_km2),
            # DeepLab stats
            "X-Oil-Spill-Present-DeepLab": "true" if deeplab_has_spill else "false",
            "X-Oil-Spill-Fraction-DeepLab": str(deeplab_fraction),
            "X-Oil-Spill-Pixels-DeepLab": str(deeplab_oil_pixels),
            "X-Oil-Spill-Total-Pixels-DeepLab": str(deeplab_total_pixels),
            "X-Oil-Spill-Area-Km2-DeepLab": str(deeplab_area_km2),
        }

        return StreamingResponse(
            io.BytesIO(buf.read()),
            media_type="image/png",
            headers=headers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper to draw Roboflow polygons onto image and return PNG buffer
def draw_aerial_overlay_image(image_path: str, rf_result: dict) -> io.BytesIO:
    # Normalize Roboflow result to a dict root
    rf_root = rf_result[0] if isinstance(rf_result, list) and rf_result else rf_result
    if not isinstance(rf_root, dict):
        rf_root = {}

    # Load original image
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    # Transparent overlay to draw polygons with alpha
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Extract detections from various Roboflow schema variants
    image_meta = (
        rf_root.get("predictions", {}).get("image")
        or rf_root.get("image")
        or rf_root.get("metadata", {}).get("image")
    )
    if isinstance(rf_root.get("predictions"), list):
        detections = rf_root["predictions"]
    elif isinstance(rf_root.get("predictions", {}).get("predictions"), list):
        detections = rf_root["predictions"]["predictions"]
    elif isinstance(rf_root.get("outputs"), list) and rf_root["outputs"]:
        detections = rf_root["outputs"][0].get("predictions", [])
    else:
        detections = []

    # Colors
    class_to_color = {
        "oil": (255, 0, 0, 100),
        "water": (0, 255, 255, 80),
        "land": (153, 76, 0, 80),
        "vegetation": (0, 153, 0, 80),
    }
    class_to_stroke = {
        "oil": (255, 0, 0, 220),
        "water": (0, 200, 200, 220),
        "land": (153, 76, 0, 220),
        "vegetation": (0, 153, 0, 220),
    }

    # If image_meta width/height differs from actual image, compute scale
    src_w = (image_meta or {}).get("width") or w
    src_h = (image_meta or {}).get("height") or h
    scale_x = w / float(src_w)
    scale_y = h / float(src_h)

    for det in detections:
        pts = det.get("points") or []
        if not pts or len(pts) < 3:
            continue
        cls = str(det.get("class") or det.get("class_name") or "").lower()
        fill = class_to_color.get(cls, (255, 165, 0, 80))
        stroke = class_to_stroke.get(cls, (255, 165, 0, 220))

        poly = []
        for p in pts:
            try:
                x = float(p.get("x", 0)) * scale_x
                y = float(p.get("y", 0)) * scale_y
                poly.append((x, y))
            except Exception:
                continue

        # Draw filled polygon then outline
        draw.polygon(poly, fill=fill)
        draw.line(poly + [poly[0]], fill=stroke, width=3)

    # Composite overlay onto original
    composited = Image.alpha_composite(img, overlay)

    # Save to buffer
    buf = io.BytesIO()
    composited.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf


@app.post("/detect/sarvsdrone")
async def detect_sar_vs_drone(file: UploadFile = File(...)):
    """Detect whether an image is SAR (satellite SAR) or aerial/drone.

    Returns JSON: { source: 'sar'|'aerial', confidence: float }
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if globals().get("sarvsdrone_model") is None:
        raise HTTPException(status_code=500, detail="SAR vs Drone model not loaded")

    try:
        # Read and preprocess similarly to the example script
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize/crop to 224x224 as the example script does
        size = (224, 224)
        try:
            img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
        except Exception:
            # Pillow older versions
            img = ImageOps.fit(img, size)

        image_array = np.asarray(img).astype(np.float32)
        # Normalize to [-1, 1]
        normalized = (image_array / 127.5) - 1.0
        input_tensor = np.expand_dims(normalized, axis=0)

        pred = globals()["sarvsdrone_model"].predict(input_tensor)

        # Interpret prediction: support sigmoid single-output or softmax two-output
        source = "aerial"
        confidence = 0.0
        try:
            pred = np.asarray(pred)
            if pred.ndim == 2 and pred.shape[1] == 2:
                # Softmax two-class
                idx = int(np.argmax(pred[0]))
                confidence = float(np.max(pred[0]))
                # Allow configurable mapping via env var SARVSDRONE_SAR_INDEX
                # This should be set to the class index (0 or 1) that corresponds to SAR
                try:
                    sar_index = int(os.getenv("SARVSDRONE_SAR_INDEX", "0"))
                except Exception:
                    sar_index = 0
                source = "sar" if idx == sar_index else "aerial"
            elif pred.ndim == 2 and pred.shape[1] == 1:
                # Sigmoid single logit/probability
                prob = float(pred[0][0])
                # If model outputs logits, convert to probability via sigmoid
                if prob < 0 or prob > 1:
                    prob = 1.0 / (1.0 + np.exp(-prob))
                confidence = prob
                # Allow inverting sigmoid semantics via env var SARVSDRONE_SIGMOID_INVERT
                invert_sig = os.getenv("SARVSDRONE_SIGMOID_INVERT", "0") in ("1", "true", "True", "yes", "y")
                if invert_sig:
                    source = "sar" if prob < 0.5 else "aerial"
                else:
                    source = "sar" if prob >= 0.5 else "aerial"
            else:
                # Fallback: flatten and pick max
                flat = pred.flatten()
                idx = int(np.argmax(flat))
                confidence = float(flat[idx])
                source = "sar" if idx == 1 else "aerial"
        except Exception:
            # On any interpretation error, fallback to aerial low confidence
            source = "aerial"
            confidence = 0.0

        return {"source": source, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/aerial")
async def predict_aerial(file: UploadFile = File(...)):
    """Run Roboflow workflow for aerial images and return a visualization PNG."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    api_url = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
    api_key = os.getenv("ROBOFLOW_API_KEY", "LxvwQNCRYOly9AOhA0ju")
    workspace_name = os.getenv("ROBOFLOW_WORKSPACE", "oilspillmaverick")
    workflow_id = os.getenv("ROBOFLOW_WORKFLOW_ID", "custom-workflow-2")

    try:
        # Persist upload to a temporary file because the SDK expects a path
        suffix = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

        result = client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            images={
                "image": tmp_path,
            },
            use_cache=True,
        )

        # Render overlay image
        buf = draw_aerial_overlay_image(tmp_path, result)

        # Cleanup temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return StreamingResponse(
            io.BytesIO(buf.read()),
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=aerial_prediction.png"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
