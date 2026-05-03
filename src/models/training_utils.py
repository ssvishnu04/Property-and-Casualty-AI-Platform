from pathlib import Path
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)


def log_classification_metrics(y_test, preds, probs=None, average="binary"):
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average=average, zero_division=0),
        "recall": recall_score(y_test, preds, average=average, zero_division=0),
        "f1": f1_score(y_test, preds, average=average, zero_division=0),
    }

    # ROC-AUC only for binary classification when probabilities are available.
    if probs is not None and len(set(y_test)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, probs)
        except Exception:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    return metrics


def log_regression_metrics(y_test, preds):
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    return metrics


def save_reports(y_test, preds, model_name, target):
    report = classification_report(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    path = Path(f"reports/model_metrics/{target}_{model_name}.txt")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(report + "\n\nConfusion Matrix:\n" + str(cm))

    mlflow.log_artifact(str(path))