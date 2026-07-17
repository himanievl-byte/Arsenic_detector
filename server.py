import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ─────────────────────── Config ────────────────────────
PORT  = 5000
DEBUG = True

# One entry per supported analyte. Add new analytes here without touching
# the rest of the code.
ANALYTE_CONFIG = {
    "chlorine": {
        "csv":       "raw_chlorine_data.csv",
        "unit":      "mg/L",
        "threshold": None,   # no regulatory flag for chlorine in this app
    },
    "arsenic": {
        "csv":       "raw_arsenic_data.csv",
        "unit":      "ppb",
        "threshold": 10.0,   # WHO guideline value for arsenic in drinking water
    },
}
DEFAULT_ANALYTE = "chlorine"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChromaSense")

app = Flask(__name__)
CORS(app)


# ─────────────────────── Dataset ───────────────────────
class ColorimetryDataset:

    def __init__(self, path: str, analyte: str, unit: str, threshold: float | None = None):
        self.path      = path
        self.analyte   = analyte
        self.unit      = unit
        self.threshold = threshold
        self.df        = pd.DataFrame()
        self.model     = None
        self.r2        = 0.0
        self.g_min     = 0.0
        self.g_max     = 255.0
        self.loaded    = False
        self.load()

    def load(self):
        if not Path(self.path).exists():
            logger.warning(f"Dataset '{self.path}' not found for '{self.analyte}'. Using demo data.")
            self._load_demo_data()
            return

        try:
            df = pd.read_csv(self.path)
            df.columns = [c.strip().lower() for c in df.columns]

            if "g" not in df.columns or "concentration" not in df.columns:
                raise ValueError("CSV must contain 'G' and 'concentration'.")

            self.df = df.dropna(subset=["g", "concentration"]).copy()

            dupes = self.df.duplicated().sum()
            if dupes:
                logger.warning(
                    f"'{self.analyte}' dataset has {dupes} exact duplicate row(s) — "
                    f"check that replicate images weren't processed twice."
                )

            self._fit_model()
            self.loaded = True

            logger.info(
                f"[{self.analyte}] Loaded {len(self.df)} rows | R²={self.r2:.4f} | "
                f"slope={self.model.coef_[0]:.6f}"
            )

        except Exception as exc:
            logger.error(f"Dataset load failed for '{self.analyte}': {exc}")
            self._load_demo_data()

    def _load_demo_data(self):
        demo_g    = [245,228,211,203]
        demo_conc = [0.5,1.0,1.5,2.0]
        self.df = pd.DataFrame({"g": demo_g, "concentration": demo_conc})
        self._fit_model()
        self.loaded = True

    def _fit_model(self):
        X = self.df[["g"]].values.astype(float)
        y = self.df["concentration"].values.astype(float)

        self.model = LinearRegression()
        self.model.fit(X, y)

        y_pred  = self.model.predict(X)
        self.r2 = float(r2_score(y, y_pred))

        self.g_min = float(self.df["g"].min())
        self.g_max = float(self.df["g"].max())

    def predict(self, g_value: float) -> dict:
        if self.model is None:
            raise RuntimeError("Model not fitted")

        concentration = float(self.model.predict([[g_value]])[0])
        concentration = max(0.0, round(concentration, 4))

        in_range   = self.g_min <= g_value <= self.g_max
        confidence = round(self.r2 * (0.95 if in_range else 0.60), 4)

        result = {
            "concentration": concentration,
            "confidence":    confidence,
            "match_id":      "LinearRegression",
            "analyte":       self.analyte.capitalize(),
            "unit":          self.unit,
            "in_calibrated_range": in_range,
        }

        if self.threshold is not None:
            exceeds = concentration > self.threshold
            result["threshold"] = self.threshold
            result["exceeds_threshold"] = exceeds
            result["message"] = (
                f"Exceeds WHO guideline of {self.threshold} {self.unit}"
                if exceeds else
                f"Within WHO guideline of {self.threshold} {self.unit}"
            )

        if not in_range:
            note = f"G value {g_value:.1f} is outside the calibrated range ({self.g_min:.1f}-{self.g_max:.1f}); result is extrapolated."
            result["message"] = (result.get("message", "") + " " + note).strip()

        return result


# Instantiate one dataset per analyte at startup
datasets: dict[str, ColorimetryDataset] = {
    name: ColorimetryDataset(
        path=cfg["csv"],
        analyte=name,
        unit=cfg["unit"],
        threshold=cfg["threshold"],
    )
    for name, cfg in ANALYTE_CONFIG.items()
}


# ─────────────────────── Helpers ───────────────────────
def extract_mean_rgb(image_bytes, x, y, w, h):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    ih, iw = img.shape[:2]

    x1 = int(max(0, min(x, iw - 1)))
    y1 = int(max(0, min(y, ih - 1)))
    x2 = int(max(x1 + 1, min(x + w, iw)))
    y2 = int(max(y1 + 1, min(y + h, ih)))

    roi = img[y1:y2, x1:x2]

    mean_bgr = cv2.mean(roi)[:3]
    b = int(round(mean_bgr[0]))
    g = int(round(mean_bgr[1]))
    r = int(round(mean_bgr[2]))

    return r, g, b


# ─────────────────────── Routes ────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    analyte = request.form.get("analyte", DEFAULT_ANALYTE).strip().lower()
    if analyte not in datasets:
        return jsonify({
            "error": f"Unknown analyte '{analyte}'. Supported: {list(datasets.keys())}"
        }), 400

    dataset = datasets[analyte]

    image_bytes = request.files["image"].read()

    roi_x      = float(request.form.get("roi_x", 0))
    roi_y      = float(request.form.get("roi_y", 0))
    roi_width  = float(request.form.get("roi_width", 0))
    roi_height = float(request.form.get("roi_height", 0))

    r, g, b = extract_mean_rgb(image_bytes, roi_x, roi_y, roi_width, roi_height)

    # ─────────────── CORE LOGIC (YOUR CHANGE) ───────────────
    offset = 255 - r
    corrected_g = g + offset

    logger.info(
        f"[{analyte}] RGB=({r},{g},{b}) | offset={offset} | corrected_G={corrected_g}"
    )

    prediction = dataset.predict(corrected_g)

    return jsonify({
        "rgb": {"r": r, "g": g, "b": b},
        "offset": offset,
        "corrected_g": corrected_g,
        **prediction
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "analytes": {
            name: {"r2": ds.r2, "loaded": ds.loaded, "rows": len(ds.df)}
            for name, ds in datasets.items()
        }
    })


# ─────────────────────── Run ───────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)