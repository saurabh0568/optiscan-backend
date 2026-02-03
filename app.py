# =========================================================
# app.py  –  Flask API for OptiScan Cataract Detection
# Deploy on Render (free).  GitHub Pages frontend POSTs here.
# =========================================================

import os, io, pickle, warnings
import cv2
import torch
import numpy as np
import albumentations as A
from torch import nn
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS          # ← allows cross-origin POST from GitHub Pages

import timm

warnings.filterwarnings("ignore")

# ─── device ──────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── paths ───────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model.pkl")

# ─── Flask app ───────────────────────────────────────────
app  = Flask(__name__)
CORS(app)                            # permit all origins (GitHub Pages URL)

# =========================================================
# CBAM
# =========================================================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.spatial(torch.cat([avg, mx], dim=1))
        return x

# =========================================================
# EfficientNet-B4 + CBAM
# =========================================================
class EfficientNet_CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=True, num_classes=0
        )
        self.cbam = CBAM(1792)

    def forward(self, x):
        x = self.backbone(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.cbam(x)
        return x.view(x.size(0), -1)

# ─── load models at startup ──────────────────────────────
print("[INFO] Loading EfficientNet-B4 + CBAM …")
feature_extractor = EfficientNet_CBAM().to(device)
feature_extractor.eval()

print("[INFO] Loading XGBoost classifier …")
with open(MODEL_PATH, "rb") as f:
    xgb_clf = pickle.load(f)

# =========================================================
# PRE-PROCESSING  (identical to training)
# =========================================================
def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def segment_lens(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

UPLOAD_TRANSFORM = A.Compose([
    A.Resize(380, 380),
    A.Normalize()
])

# =========================================================
# FUNDUS VALIDATION
# =========================================================
def is_fundus_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mean_val = np.mean(gray)
    if mean_val < 40 or mean_val > 220:
        return False

    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (img_bgr.shape[0] * img_bgr.shape[1])
    if area_ratio < 0.2:
        return False

    b, g, r = cv2.split(img_bgr)
    if np.mean(r) < np.mean(g) or np.mean(r) < np.mean(b):
        return False

    return True

# =========================================================
# PREDICTION PIPELINE
# =========================================================
def predict(img_bgr: np.ndarray) -> dict:
    if not is_fundus_image(img_bgr):
        return {"valid": False,
                "error": "Not a valid fundus (retinal) image. "
                         "Please upload a proper retinal fundus photograph."}

    img = apply_clahe(img_bgr)
    img = segment_lens(img)
    img = UPLOAD_TRANSFORM(image=img)["image"]
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        features = feature_extractor(tensor).cpu().numpy()

    prob_cataract = float(xgb_clf.predict_proba(features)[0, 1])
    label = "Cataract" if prob_cataract >= 0.5 else "Normal"

    return {
        "valid":           True,
        "prediction":      label,
        "cataract_prob":   round(prob_cataract, 4),
        "normal_prob":     round(1 - prob_cataract, 4),
        "threshold":       0.50
    }

# =========================================================
# ROUTES
# =========================================================
@app.route("/health")                # Render health-check
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": f"Cannot decode image: {str(e)}"}), 400

    result = predict(img_bgr)
    return jsonify(result)

# =========================================================
# ENTRY POINT  (used only for local dev; Render uses gunicorn)
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
