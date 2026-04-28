# Marketing ROI API

Simple FastAPI for Sales predictions, using the trained model artifacts from src/models.

## Install & Run

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run API
python main.py
```

API will be at: `http://localhost:8000`

## Endpoints

### POST `/predict`
Single prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tv": 120, "radio": 25, "social_media": 80, "influencer": "Macro"}'
```

### POST `/predict-batch`
Batch predictions
```bash
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{"items": [{"tv": 120, "radio": 25, "social_media": 80, "influencer": "Macro"}]}'
```

### GET `/health`
Health check

### GET `/model-info`
Model information

### GET `/docs`
Interactive documentation (Swagger UI)

## Setup

1. Train the model via the script: `python src/models/train.py`
2. Ensure these files exist:
  - `src/models/marketing_model.joblib`
  - `src/models/preprocessor.joblib`
3. Start API: `python main.py`
4. Test: http://localhost:8000/docs
