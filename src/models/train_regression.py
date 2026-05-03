import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split

from src.models.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.models.model_candidates import get_regression_models
from src.models.hyperparameter_tuning import tune_model
from src.models.training_utils import log_regression_metrics


def train_regression(df, target, target_name):
    df = df.dropna(subset=[target]).copy()

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    results = []
    best_score = float("inf")
    champion = None
    champion_name = None

    models = get_regression_models()

    for name, model in models.items():
        print(f"Training {target_name} model: {name}")

        with mlflow.start_run(run_name=f"{target_name}_{name}"):
            tuned_model, best_params = tune_model(
                model_name=name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                problem_type="regression",
                scoring="neg_root_mean_squared_error",
            )

            pipeline = tuned_model
            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)

            metrics = log_regression_metrics(y_test, preds)

            mlflow.log_param("target", target)
            mlflow.log_param("target_name", target_name)
            mlflow.log_param("model_name", name)

            if best_params:
                mlflow.log_params(best_params)

            results.append(
                {
                    "target": target_name,
                    "model": name,
                    **metrics,
                }
            )

            if metrics["rmse"] < best_score:
                best_score = metrics["rmse"]
                champion = pipeline
                champion_name = name

    results_df = pd.DataFrame(results)
    results_df["champion_metric"] = "rmse"
    results_df["is_champion"] = results_df["model"] == champion_name

    print(f"Champion for {target_name}: {champion_name} using rmse={best_score:.4f}")

    return results_df, champion, champion_name, "rmse", best_score