# backend/model_loader.py

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


class ModelLoader:
    def __init__(self):
        self.hedonic_model = joblib.load(MODEL_DIR / "lgbm_hedonic.pkl")
        self.anomaly_model = joblib.load(MODEL_DIR / "iso_forest.pkl")
        self.classifier_model = joblib.load(MODEL_DIR / "lgbm_classifier.pkl")
        self.tier_encoder = joblib.load(MODEL_DIR / "tier_encoder.pkl")


models = ModelLoader()