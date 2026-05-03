import json
import joblib
from pathlib import Path


MODEL_ARTIFACT_DIR = Path("data/model_artifacts")
METADATA_PATH = MODEL_ARTIFACT_DIR / "champion_metadata.json"


def save_champion(model, name, model_name, metric, score):
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_ARTIFACT_DIR / f"{name}_champion.pkl"

    # Save only model pipeline in pkl
    joblib.dump(model, model_path)

    metadata = load_existing_metadata()

    metadata[name] = {
        "model_name": model_name,
        "metric": metric,
        "score": float(score),
        "artifact_path": str(model_path),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return model_path


def load_existing_metadata():
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)

    return {}