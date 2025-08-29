"""Evaluate LightGBM models on the test set (version-proof plotting)."""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_curve,
    auc,
)
from urllib.parse import urlparse
import boto3
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import lightgbm as lgb
import json
import logging
import os
import pathlib
import pickle
import tarfile
import sys
import subprocess

# Pin only what we truly need here; avoid touching scikit-learn version.
subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm==3.3.5"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

matplotlib.use("Agg")


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
]


def _load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
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
    missing_cols = [col for col in FEATURES if col not in subset.columns]
    if missing_cols:
        logging.warning("Adding missing feature columns with default 0: %s", missing_cols)
        for col in missing_cols:
            subset[col] = 0
    X = subset[FEATURES] if not subset.empty else subset
    y = subset[TARGET] if TARGET in subset.columns else pd.Series(dtype=int)
    return X, y


def _safe_metric(fn, *args, metric_name: str = "", default=None, **kwargs):
    """Run a metric function; if it errors (e.g., single-class y), log and return default."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logging.warning("Metric '%s' failed: %s", metric_name or getattr(fn, "__name__", str(fn)), e)
        return default


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Unpack model artifacts
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    # Load test data
    test_path = "/opt/ml/processing/test/test.csv"
    test_df = _load_dataset(test_path)

    overall_true = []
    overall_pred = []
    metrics = {}

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Optional S3 upload target for plots / report
    plots_s3_uri = os.environ.get("EVAL_PLOTS_S3_URI")
    if plots_s3_uri:
        parsed_uri = urlparse(plots_s3_uri)
        s3_client = boto3.client("s3")
        s3_bucket = parsed_uri.netloc
        s3_prefix = parsed_uri.path.lstrip("/")
    else:
        s3_client = None
        s3_bucket = s3_prefix = None

    for horizon in HORIZONS:
        model_file = f"model_horizon_{horizon}w.pkl"
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        X_test, y_test = get_data_for_horizon(test_df, horizon)

        # Skip empty horizons early
        if X_test.empty or y_test.empty:
            logging.warning("No rows for horizon %s; skipping plots and metrics.", horizon)
            continue

        # Predictions
        # LightGBM sklearn API: predict -> class labels; predict_proba -> probs
        # Ensure integer labels for safety.
        y_pred = model.predict(X_test)
        # Some environments might return floats; coerce to ints if binary probs were returned.
        if y_pred.ndim > 1 or set(pd.Series(y_pred).unique()) - {0, 1}:
            # If it's not {0,1}, derive from probabilities
            y_proba_tmp = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba_tmp >= 0.5).astype(int)
        y_pred = pd.Series(y_pred).astype(int).to_list()

        # Probabilities for metrics/curves
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics (robust to single-class cases)
        acc = _safe_metric(accuracy_score, y_test, y_pred, metric_name="accuracy", default=None)
        prec = _safe_metric(precision_score, y_test, y_pred, metric_name="precision", default=None)
        rec = _safe_metric(recall_score, y_test, y_pred, metric_name="recall", default=None)
        f1 = _safe_metric(f1_score, y_test, y_pred, metric_name="f1", default=None)
        roc_auc = _safe_metric(roc_auc_score, y_test, y_proba, metric_name="roc_auc", default=None)
        pr_auc = _safe_metric(
            average_precision_score, y_test, y_proba, metric_name="pr_auc", default=None
        )

        metrics[str(horizon)] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

        overall_true.extend(pd.Series(y_test).astype(int).to_list())
        overall_pred.extend(y_pred)

        # ---------- Confusion Matrix (version-proof) ----------
        try:
            cm_disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
            fig_cm = cm_disp.figure_
        except AttributeError:
            cm = confusion_matrix(y_test, y_pred)
            cm_disp = ConfusionMatrixDisplay(cm)
            fig_cm, ax_cm = plt.subplots()
            cm_disp.plot(ax=ax_cm)

        cm_path = os.path.join(output_dir, f"confusion_matrix_horizon_{horizon}.png")
        fig_cm.savefig(cm_path, bbox_inches="tight")
        plt.close(fig_cm)
        if s3_client:
            s3_client.upload_file(
                cm_path, s3_bucket, os.path.join(s3_prefix, os.path.basename(cm_path))
            )

        # ---------- ROC Curve (version-proof) ----------
        try:
            roc_disp = RocCurveDisplay.from_predictions(y_test, y_proba)
            fig_roc = roc_disp.figure_
        except AttributeError:
            # Manual computation
            try:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc_manual = auc(fpr, tpr)
                roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_manual)
                fig_roc, ax_roc = plt.subplots()
                roc_disp.plot(ax=ax_roc)
            except Exception as e:
                logging.warning("ROC computation failed for horizon %s: %s", horizon, e)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.set_title(f"ROC unavailable for horizon {horizon}")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")

        roc_path = os.path.join(output_dir, f"roc_curve_horizon_{horizon}.png")
        fig_roc.savefig(roc_path, bbox_inches="tight")
        plt.close(fig_roc)
        if s3_client:
            s3_client.upload_file(
                roc_path, s3_bucket, os.path.join(s3_prefix, os.path.basename(roc_path))
            )

        # ---------- Feature Importance ----------
        ax = lgb.plot_importance(
            model.booster_, importance_type="gain", max_num_features=len(FEATURES)
        )
        fig_fi = ax.figure
        fi_path = os.path.join(output_dir, f"feature_importance_horizon_{horizon}.png")
        fig_fi.savefig(fi_path, bbox_inches="tight")
        plt.close(fig_fi)
        if s3_client:
            s3_client.upload_file(
                fi_path, s3_bucket, os.path.join(s3_prefix, os.path.basename(fi_path))
            )

    # Overall metrics
    if overall_true and overall_pred:
        overall_accuracy = _safe_metric(
            accuracy_score, overall_true, overall_pred, metric_name="overall_accuracy", default=None
        )
    else:
        overall_accuracy = None
        logging.warning("No evaluations were run; overall metrics unavailable.")

    report_dict = {
        "classification_metrics": {"accuracy": {"value": overall_accuracy}},
        "per_horizon_metrics": metrics,
    }

    # Persist evaluation report
    evaluation_path = os.path.join(output_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    if s3_client:
        s3_client.upload_file(
            evaluation_path,
            s3_bucket,
            os.path.join(s3_prefix, os.path.basename(evaluation_path)),
        )
