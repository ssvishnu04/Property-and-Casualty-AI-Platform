import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.models.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.models.model_candidates import get_classification_models
from src.models.hyperparameter_tuning import tune_model
from src.models.training_utils import log_classification_metrics, save_reports


def train_classification(df, target, target_name, metric, average):
    """
    Trains multiple classification models for a given target.

    Used for:
    - severity classification
    - fraud / suspicious claim classification

    It:
    1. Drops missing target values
    2. Encodes labels safely
    3. Splits train/test
    4. Trains candidate models
    5. Applies hyperparameter tuning where available
    6. Logs metrics to MLflow
    7. Selects champion model
    """
    df = df.dropna(subset=[target]).copy()

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_raw = df[target]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    print(f"\nTraining target: {target_name}")
    print(f"Original labels: {list(encoder.classes_)}")
    print(f"Encoded labels: {sorted(set(y))}")

    if len(set(y)) < 2:
        raise ValueError(
            f"Target {target_name} has only one class after encoding. "
            "Generate more data before training."
        )

    class_counts = pd.Series(y).value_counts()
    can_stratify = class_counts.min() >= 2
    stratify_value = y if can_stratify else None

    if not can_stratify:
        print(
            f"Warning: Not enough samples per class for stratified split on {target_name}. "
            "Using regular train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value,
    )

    results = []
    best_score = -1
    champion = None
    champion_name = None

    models = get_classification_models()

    for name, model in models.items():
        print(f"Training {target_name} model: {name}")

        with mlflow.start_run(run_name=f"{target_name}_{name}"):
            mlflow.log_param("target", target)
            mlflow.log_param("target_name", target_name)
            mlflow.log_param("model_name", name)
            mlflow.log_param("original_labels", ",".join(map(str, encoder.classes_)))
            mlflow.log_param("encoded_labels", ",".join(map(str, sorted(set(y)))))

            scoring = "f1_macro" if average == "macro" else "recall"

            tuned_model, best_params = tune_model(
                model_name=name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                problem_type="classification",
                scoring=scoring,
            )

            pipeline = tuned_model
            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)

            probs = None
            if hasattr(pipeline, "predict_proba"):
                predicted_probabilities = pipeline.predict_proba(X_test)

                if predicted_probabilities.shape[1] == 2:
                    probs = predicted_probabilities[:, 1]

            metrics = log_classification_metrics(
                y_test,
                preds,
                probs,
                average=average,
            )

            save_reports(y_test, preds, name, target_name)

            if best_params:
                mlflow.log_params(best_params)

            results.append(
                {
                    "target": target_name,
                    "model": name,
                    **metrics,
                }
            )

            score = metrics.get(metric, 0)

            if score > best_score:
                best_score = score
                champion = pipeline
                champion_name = name

    results_df = pd.DataFrame(results)
    results_df["champion_metric"] = metric
    results_df["is_champion"] = results_df["model"] == champion_name

    print(f"Champion for {target_name}: {champion_name} using {metric}={best_score:.4f}")

    return results_df, champion, champion_name, metric, best_score