"""FastAPI backend for the MARCO web app - runs the trained model on each request.

Serves the upload page and the /predict endpoint. The model is loaded once at startup (the
weights never change between requests), but every prediction runs a fresh forward pass over
the uploaded image - nothing is precomputed or hard-coded (CLAUDE.md rule 10). Uses the Final
ANN (layer4+head unfrozen), the best model by the primary metric - see
docs/ann_initial_training.md and experiments/log.csv.
"""

import base64
import io
import json
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from matplotlib import cm
from PIL import Image, UnidentifiedImageError

from marco.data import CLASS_ORDER, build_transforms
from marco.model import build_model
from marco.xai import GradCAM

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "experiments" / "models" / "ann_final.pt"
HISTORY_PATH = PROJECT_ROOT / "experiments" / "models" / "ann_final_history.json"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

EXPLANATION = (
    "This score is the model's output probability for each class, not a calibrated or "
    "guaranteed confidence - it reflects patterns learned from the training data, not proof "
    "of correctness. Always inspect flagged drops visually before acting on a prediction, "
    "especially for Crystals."
)
LIMITATIONS = (
    "Trained on MARCO dataset images from a fixed set of imaging setups. Images from unseen "
    "microscopes, lighting conditions, or plate types may reduce accuracy. The model has no "
    "way to say 'I don't recognise this' - testing showed it can confidently misclassify "
    "images that don't resemble a crystallisation drop at all (e.g. random noise scored 56% "
    "Crystals). Only use this on actual drop images."
)

app = FastAPI(title="MARCO Crystallisation Triage")

params = json.loads(HISTORY_PATH.read_text())["params"]
model = build_model(params["backbone"])
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
transform = build_transforms(train=False)
grad_cam = GradCAM(model)

GRADCAM_EXPLANATION = (
    "Red/yellow shows where the model's gradients were most sensitive for this prediction - "
    "not proof of the true reason for the prediction, just supporting evidence of what "
    "influenced it. See notebooks/04_xai.ipynb for method comparisons and sanity checks."
)


def make_overlay(img: Image.Image, heatmap) -> str:
    """Blend a Grad-CAM heatmap over the (resized) input image, return as a data: URL."""
    img_resized = img.resize((224, 224))
    colored = (cm.jet(heatmap)[:, :, :3] * 255).astype("uint8")
    heat_img = Image.fromarray(colored).convert("RGB")
    overlay = Image.blend(img_resized, heat_img, alpha=0.45)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Could not read this file as an image. Please upload a JPEG or PNG.",
        )

    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    scores = {cls: round(float(p), 4) for cls, p in zip(CLASS_ORDER, probs)}
    predicted = max(scores, key=scores.get)

    heatmap = grad_cam(tensor, CLASS_ORDER.index(predicted))
    gradcam_overlay = make_overlay(img, heatmap)

    return JSONResponse({
        "predicted_class": predicted,
        "class_scores": scores,
        "explanation": EXPLANATION,
        "limitations": LIMITATIONS,
        "gradcam_overlay": gradcam_overlay,
        "gradcam_explanation": GRADCAM_EXPLANATION,
    })
