from sklearn.model_selection import RandomizedSearchCV

from src.models.preprocessing import build_pipeline


def get_classification_search_spaces():
    return {
        "RandomForestClassifier": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [6, 8, 10, 12],
            "model__min_samples_split": [5, 10, 20],
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features": ["sqrt", "log2"],
        },
        "GradientBoostingClassifier": {
            "model__n_estimators": [100, 150, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        },
        "XGBClassifier": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.8, 0.9, 1.0],
            "model__reg_alpha": [0, 0.1],
            "model__reg_lambda": [1.0, 2.0],
        },
    }


def get_regression_search_spaces():
    return {
        "RandomForestRegressor": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [6, 8, 10, 12],
            "model__min_samples_split": [5, 10, 20],
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features": ["sqrt", "log2"],
        },
        "GradientBoostingRegressor": {
            "model__n_estimators": [100, 150, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        },
        "XGBRegressor": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.8, 0.9, 1.0],
            "model__reg_alpha": [0, 0.1],
            "model__reg_lambda": [1.0, 2.0],
        },
    }


def tune_model(
    model_name,
    model,
    X_train,
    y_train,
    problem_type,
    scoring,
    n_iter=8,
    cv=3,
):
    search_spaces = (
        get_classification_search_spaces()
        if problem_type == "classification"
        else get_regression_search_spaces()
    )

    if model_name not in search_spaces:
        return build_pipeline(model), None

    pipeline = build_pipeline(model)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_spaces[model_name],
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=42,
        n_jobs=-1,
        error_score="raise",
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_