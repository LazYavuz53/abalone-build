"""Train LightGBM models per forecast horizon."""

import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm==3.3.5"])
import lightgbm as lgb

import os
import pickle

import pandas as pd
from lightgbm import early_stopping, log_evaluation

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
MODEL_DIR = "/opt/ml/model"
PREDICTION_DIR = "/opt/ml/output/predictions"

# column names as produced by preprocess.py
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
    """Load dataset saved by preprocess step."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    # cast categorical columns to category dtype
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
    cat_cols = X.select_dtypes(include=["category"]).columns
    return X, y, cat_cols


def build_model():
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=64,
        learning_rate=0.05,
        n_estimators=2000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )


def train_and_evaluate(horizon, train_df, val_df):
    X_train, y_train, cat_cols = get_data_for_horizon(train_df, horizon)
    X_val, y_val, _ = get_data_for_horizon(val_df, horizon)
    model = build_model()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "average_precision"],
        categorical_feature=list(cat_cols),
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(100)],
    )

    # save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"model_horizon_{horizon}w.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # save validation predictions for inspection
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    preds_df = X_val.copy()
    preds_df["TrueLabel"] = y_val.values
    preds_df["PredictedLabel"] = y_pred
    preds_df["PredictedProb"] = y_proba
    preds_df.to_csv(
        os.path.join(PREDICTION_DIR, f"predictions_horizon_{horizon}w.csv"),
        index=False,
    )


def main():
    train_path = "/opt/ml/input/data/train/train.csv"
    val_path = "/opt/ml/input/data/validation/validation.csv"
    train_df = _load_dataset(train_path)
    val_df = _load_dataset(val_path)

    for horizon in HORIZONS:
        train_and_evaluate(horizon, train_df, val_df)


if __name__ == "__main__":
    main()