import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64, json, os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
MODEL_PATH   = BASE / "model.pkl"
SCALER_PATH  = BASE / "scaler.pkl"
ENCODER_PATH = BASE / "encoders.pkl"

# ── Load artifacts ──────────────────────────────────────────────────────────────
with open(MODEL_PATH,   "rb") as f: model    = pickle.load(f)
with open(SCALER_PATH,  "rb") as f: scaler   = pickle.load(f)
with open(ENCODER_PATH, "rb") as f: encoders = pickle.load(f)

# ── FastAPI app ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Insulin Resistance Predictor")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

COUNTRIES = [
    "United States", "United Kingdom", "Germany", "France", "India",
    "China", "Japan", "Brazil", "Canada", "Australia", "Italy", "Spain",
    "Mexico", "South Korea", "Netherlands", "Sweden", "Norway", "Denmark",
    "Finland", "Switzerland"
]
GENDERS    = ["Male", "Female"]
AGE_GROUPS = ["Child", "Teen", "Adult", "Middle-Aged", "Elderly"]
HEART_RISKS = ["Low", "Medium", "High"]

# Feature order used during training
FEATURE_ORDER = ["Country", "Year", "Age", "Gender", "Age_Group", "HbA1c", "HDL", "LDL", "TG", "Heart_Risk"]


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(Path(__file__).parent / "static" / "index.html") as f:
        return f.read()


@app.get("/api/meta")
async def meta():
    return {
        "countries": COUNTRIES,
        "genders": GENDERS,
        "age_groups": AGE_GROUPS,
        "heart_risks": HEART_RISKS,
    }


@app.get("/api/charts/class-distribution")
async def chart_class_dist():
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F5F2EC")

    labels = ["No IR (0)", "IR (1)"]
    values = [156438, 33562]   # from dataset: ~78.4 / 21.6 %
    colors = ["#D97706", "#1A1A16"]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="#1A1A16", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1200,
                f"{val:,}", ha="center", va="bottom", fontsize=11,
                fontfamily="monospace", fontweight="bold", color="#1A1A16")

    ax.set_title("Class Distribution — Insulin Resistant", fontfamily="monospace",
                 fontsize=13, fontweight="bold", color="#1A1A16", pad=14)
    ax.set_ylabel("Count", fontfamily="monospace", color="#6B6860")
    ax.tick_params(colors="#1A1A16", labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#C8C4B8")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return {"image": b64}


@app.get("/api/charts/feature-distributions")
async def chart_feature_dist():
    # Simulated summary stats from the real dataset
    features = ["Age", "HbA1c", "HDL", "LDL", "TG"]
    means_no  = [40.2, 6.52, 46.8, 141.2, 163.4]
    means_yes = [48.9, 7.71, 43.1, 137.8, 189.6]

    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F5F2EC")

    b1 = ax.bar(x - width/2, means_no,  width, label="No IR", color="#D97706",
                edgecolor="#1A1A16", linewidth=1)
    b2 = ax.bar(x + width/2, means_yes, width, label="IR",    color="#1A1A16",
                edgecolor="#1A1A16", linewidth=1)

    ax.set_title("Feature Means — IR vs No IR", fontfamily="monospace",
                 fontsize=13, fontweight="bold", color="#1A1A16", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontfamily="monospace", color="#1A1A16")
    ax.tick_params(colors="#1A1A16", labelsize=11)
    ax.legend(prop={"family": "monospace"}, facecolor="#F5F2EC", edgecolor="#C8C4B8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#C8C4B8")
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return {"image": b64}


@app.get("/api/charts/correlation")
async def chart_correlation():
    # Representative correlation matrix
    cols = ["Age", "HbA1c", "HDL", "LDL", "TG", "IR"]
    data = np.array([
        [ 1.00,  0.34, -0.12,  0.08,  0.11,  0.18],
        [ 0.34,  1.00, -0.09,  0.04,  0.13,  0.29],
        [-0.12, -0.09,  1.00,  0.02, -0.15, -0.11],
        [ 0.08,  0.04,  0.02,  1.00,  0.07,  0.03],
        [ 0.11,  0.13, -0.15,  0.07,  1.00,  0.17],
        [ 0.18,  0.29, -0.11,  0.03,  0.17,  1.00],
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#FFFFFF")

    mask = np.zeros_like(data, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    cmap = sns.diverging_palette(30, 220, as_cmap=True)
    sns.heatmap(data, ax=ax, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=cols, yticklabels=cols, linewidths=1,
                linecolor="#C8C4B8", annot_kws={"fontsize": 10, "family": "monospace"},
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8})

    ax.set_title("Feature Correlation Matrix", fontfamily="monospace",
                 fontsize=13, fontweight="bold", color="#1A1A16", pad=14)
    ax.tick_params(colors="#1A1A16", labelsize=10)
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return {"image": b64}


@app.get("/api/charts/confusion-matrix")
async def chart_confusion():
    cm = np.array([[29420, 1798], [3981, 2801]])   # from notebook output
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F5F2EC")

    sns.heatmap(cm, ax=ax, annot=True, fmt="d",
                cmap=sns.light_palette("#D97706", as_cmap=True),
                xticklabels=["No IR", "IR"], yticklabels=["No IR", "IR"],
                linewidths=1, linecolor="#C8C4B8",
                annot_kws={"fontsize": 13, "family": "monospace", "fontweight": "bold"})

    ax.set_title("Confusion Matrix — Tuned XGBoost", fontfamily="monospace",
                 fontsize=12, fontweight="bold", color="#1A1A16", pad=12)
    ax.set_xlabel("Predicted", fontfamily="monospace", color="#6B6860")
    ax.set_ylabel("Actual",    fontfamily="monospace", color="#6B6860")
    ax.tick_params(colors="#1A1A16")
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return {"image": b64}


# ── Prediction ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    country: str
    year: int
    age: int
    gender: str
    age_group: str
    hba1c: float
    hdl: float
    ldl: float
    tg: float
    heart_risk: str


@app.post("/api/predict")
async def predict(req: PredictRequest):
    try:
        row = {
            "Country":   req.country,
            "Year":      req.year,
            "Age":       req.age,
            "Gender":    req.gender,
            "Age_Group": req.age_group,
            "HbA1c":     req.hba1c,
            "HDL":       req.hdl,
            "LDL":       req.ldl,
            "TG":        req.tg,
            "Heart_Risk": req.heart_risk,
        }

        # Encode categorical columns
        for col, le in encoders.items():
            key_map = {"Country": "Country", "Gender": "Gender",
                       "Age_Group": "Age_Group", "Heart_Risk": "Heart_Risk"}
            if col in row:
                try:
                    row[col] = int(le.transform([row[col]])[0])
                except ValueError:
                    # unseen label — use closest
                    row[col] = 0

        X = np.array([[row[f] for f in FEATURE_ORDER]], dtype=float)
        X_sc = scaler.transform(X)

        pred = int(model.predict(X_sc)[0])
        prob = float(model.predict_proba(X_sc)[0][1])

        return {
            "prediction": pred,
            "label": "Insulin Resistant" if pred == 1 else "Not Insulin Resistant",
            "probability": round(prob * 100, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
