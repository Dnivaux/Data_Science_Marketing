from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="Marketing ROI API",
    version="1.0.0",
    description="Simple API for marketing ROI predictions"
)


class PredictRequest(BaseModel):
    tv: float
    radio: float
    social_media: float
    influencer: str


class PredictBatchRequest(BaseModel):
    items: list[PredictRequest]


# Global variables for model and preprocessor
model = None
preprocessor = None


@app.on_event("startup")
def load_model_on_startup():
    """Load model and scaler on startup"""
    global model, preprocessor

    models_dir = Path(__file__).resolve().parents[1] / "src" / "models"

    # Try to load saved model and preprocessor
    try:
        model_path = models_dir / "marketing_model.joblib"
        preprocessor_path = models_dir / "preprocessor.joblib"

        if model_path.exists() and preprocessor_path.exists():
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            print("✓ Model and preprocessor loaded")
        else:
            print("⚠️  Model or preprocessor files not found")
            model = None
            preprocessor = None
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
def read_root():
    """API root"""
    return {
        "name": "Marketing ROI API",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    """
    Predict sales for marketing campaign

    - **tv**: TV spend
    - **radio**: Radio spend
    - **social_media**: Social Media spend
    - **influencer**: Influencer category (e.g., Macro, Micro, Nano, Mega)
    """

    if model is None or preprocessor is None:
        return {"error": "Model or preprocessor not loaded"}

    # Prepare features in expected order
    features = pd.DataFrame([
        {
            "TV": payload.tv,
            "Radio": payload.radio,
            "Social Media": payload.social_media,
            "Influencer": payload.influencer,
        }
    ])

    # Preprocess and predict
    try:
        features_preprocessed = preprocessor.transform(features)
        prediction = float(model.predict(features_preprocessed)[0])

        return {
            "prediction": round(prediction, 2),
            "tv": payload.tv,
            "radio": payload.radio,
            "social_media": payload.social_media,
            "influencer": payload.influencer
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict-batch")
def predict_batch(payload: PredictBatchRequest):
    """
    Predict for multiple campaigns

    data: List of dicts with keys: tv, radio, social_media, influencer
    """

    if model is None or preprocessor is None:
        return {"error": "Model or preprocessor not loaded"}

    results = []

    for item in payload.items:
        try:
            features = pd.DataFrame([
                {
                    "TV": item.tv,
                    "Radio": item.radio,
                    "Social Media": item.social_media,
                    "Influencer": item.influencer,
                }
            ])

            features_preprocessed = preprocessor.transform(features)
            prediction = float(model.predict(features_preprocessed)[0])

            results.append({
                "prediction": round(prediction, 2),
                "tv": item.tv,
                "radio": item.radio,
                "social_media": item.social_media,
                "influencer": item.influencer,
            })
        except Exception as e:
            results.append({
                "error": str(e),
                "tv": item.tv,
                "radio": item.radio,
                "social_media": item.social_media,
                "influencer": item.influencer,
            })
    
    return {"results": results}


@app.get("/model-info")
def model_info():
    """Get model information"""
    return {
        "model_type": type(model).__name__ if model else "None",
        "features": ["TV", "Radio", "Social Media", "Influencer"],
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
