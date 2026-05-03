import mlflow
import pandas as pd

from src.models.features import (
    GOLD_FEATURES_PATH,
    SEVERITY_TARGET,
    FRAUD_TARGET,
    RESERVE_TARGET,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    METRICS_REPORT_DIR,
)
from src.models.train_classification import train_classification
from src.models.train_regression import train_regression
from src.models.model_selection import save_champion


def run():
    print("Starting full ML pipeline...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(GOLD_FEATURES_PATH)

    severity_results, severity_model, severity_model_name, severity_metric, severity_score = train_classification(
        df=df,
        target=SEVERITY_TARGET,
        target_name="severity",
        metric="f1",
        average="macro",
    )

    fraud_results, fraud_model, fraud_model_name, fraud_metric, fraud_score = train_classification(
        df=df,
        target=FRAUD_TARGET,
        target_name="fraud",
        metric="recall",
        average="binary",
    )

    reserve_results, reserve_model, reserve_model_name, reserve_metric, reserve_score = train_regression(
        df=df,
        target=RESERVE_TARGET,
        target_name="reserve",
    )

    comparison_df = pd.concat(
        [severity_results, fraud_results, reserve_results],
        ignore_index=True,
    )

    METRICS_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_path = METRICS_REPORT_DIR / "model_comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)

    s_path = save_champion(
        model=severity_model,
        name="severity",
        model_name=severity_model_name,
        metric=severity_metric,
        score=severity_score,
    )

    f_path = save_champion(
        model=fraud_model,
        name="fraud",
        model_name=fraud_model_name,
        metric=fraud_metric,
        score=fraud_score,
    )

    r_path = save_champion(
        model=reserve_model,
        name="reserve",
        model_name=reserve_model_name,
        metric=reserve_metric,
        score=reserve_score,
    )

    print("\nModels saved:")
    print(s_path)
    print(f_path)
    print(r_path)

    print(f"\nModel comparison saved to: {comparison_path}")

    print("\nChampion models:")
    print(
        comparison_df[comparison_df["is_champion"] == True][
            ["target", "model", "champion_metric"]
        ]
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    run()