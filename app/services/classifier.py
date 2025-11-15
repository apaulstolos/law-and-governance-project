from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from app.services.azure_client import embed_texts

MODEL_PATH = Path("models/classifier.joblib")


class CommentClassifier:
    def __init__(self) -> None:
        self.pipeline = None
        if MODEL_PATH.exists():
            self.pipeline = joblib.load(MODEL_PATH)

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def predict(self, text: str) -> Optional[dict]:
        if not self.pipeline:
            return None
        embedding = embed_texts([text])[0]
        proba = self.pipeline.predict_proba(np.array([embedding]))[0]
        label_idx = int(np.argmax(proba))
        label = self.pipeline.classes_[label_idx]
        return {
            "label": label,
            "probability": float(proba[label_idx]),
        }


classifier = CommentClassifier()
