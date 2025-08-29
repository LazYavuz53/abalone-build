"""Evaluate LightGBM models on the test set."""
import json
import logging
import os
import pathlib
import pickle
import tarfile

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

FEATURES = [
    "Supplier",
    "HierarchyLevel1",
    "HierarchyLevel2",
    "DIorDOM",
    "Seasonal",
    "SpringSummer",
    "Status",
    "SalePriceIncVAT",
    "ForecastPerWeek",
    "ActualsPerWeek",
    "WeeksOut",
]
TARGET = "DiscontinuedTF"
HORIZONS = [-12, -8, -4]

COLUMNS = [
    "DiscontinuedTF",
    "CatEdition",
    "SpringSummer",
    "ProductKey",
    "WeeksOut",
    "Status",
    "SalePriceIncVAT",
    "ForecastPerWeek",
    "ActualsPerWeek",
    "Fcast_to_Actual",
    "Supplier",
    "HierarchyLevel1",
    "HierarchyLevel2",
    "DIorDOM",
    "Seasonal",
]


def _load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    for col in [
        "Supplier",
        "HierarchyLevel1",
        "HierarchyLevel2",
        "DIorDOM",
        "Seasonal",
        "SpringSummer",
        "Status",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def get_data_for_horizon(df: pd.DataFrame, horizon: int):
    subset = df[df["WeeksOut"] == horizon].copy()
    X = subset[FEATURES]
    y = subset[TARGET]
    return X, y


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    test_path = "/opt/ml/processing/test/test.csv"
    test_df = _load_dataset(test_path)

    overall_true = []
    overall_pred = []
    metrics = {}

    for horizon in HORIZONS:
        model_file = f"model_horizon_{horizon}w.pkl"
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        X_test, y_test = get_data_for_horizon(test_df, horizon)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics[str(horizon)] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
        }
        overall_true.extend(y_test)
        overall_pred.extend(y_pred)

    overall_accuracy = accuracy_score(overall_true, overall_pred)
    report_dict = {
        "classification_metrics": {"accuracy": {"value": overall_accuracy}},
        "per_horizon_metrics": metrics,
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = os.path.join(output_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))