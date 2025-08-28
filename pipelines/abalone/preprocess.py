"""Preprocess product and catalogue data for model training."""
import argparse
import logging
import os
import pathlib
from typing import List

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def to_bool01(s: pd.Series) -> pd.Series:
    """Normalize various boolean-like strings to 0/1 integers."""
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "t", "yes", "y"])
    ).astype("int8")


def check_duplicates(df: pd.DataFrame, cols: List[str], name: str) -> None:
    """Print information about duplicated rows by specified columns."""
    dup = df.duplicated(subset=cols, keep=False)
    n = int(dup.sum())
    if n:
        logger.warning("%s: found %d duplicate rows by %s", name, n, cols)
        logger.warning("Sample:\n%s", df.loc[dup].head())
    else:
        logger.info("%s: no duplicates by %s", name, cols)


def missingness(df: pd.DataFrame, name: str) -> None:
    """Log the percentage of missing values in each column."""
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    logger.info("Missingness in %s:\n%s", name, miss.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--products-data", type=str, required=True)
    parser.add_argument("--catalogue-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    output_dir = f"{base_dir}/output"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    for sub in ["train", "validation", "test"]:
        pathlib.Path(f"{base_dir}/{sub}").mkdir(parents=True, exist_ok=True)

    def download_from_s3(uri: str, local_path: str) -> None:
        bucket = uri.split("/")[2]
        key = "/".join(uri.split("/")[3:])
        boto3.resource("s3").Bucket(bucket).download_file(key, local_path)

    prod_path = f"{base_dir}/ProductDetails.csv"
    cat_path = f"{base_dir}/CatalogueDiscontinuation.csv"
    download_from_s3(args.products_data, prod_path)
    download_from_s3(args.catalogue_data, cat_path)

    prod_dtypes = {
        "ProductKey": "Int64",
        "Supplier": "Int64",
        "HierarchyLevel1": "Int64",
        "HierarchyLevel2": "Int64",
        "DIorDOM": "string",
        "Seasonal": "string",
    }

    cat_dtypes = {
        "CatEdition": "Int64",
        "SpringSummer": "string",
        "ProductKey": "Int64",
        "WeeksOut": "Int64",
        "Status": "string",
        "SalePriceIncVAT": "float",
        "ForecastPerWeek": "float",
        "ActualsPerWeek": "float",
        "DiscontinuedTF": "string",
    }

    prod = pd.read_csv(prod_path, dtype=prod_dtypes)
    cat = pd.read_csv(cat_path, dtype=cat_dtypes)
    logger.info("Raw ProductDetails:\n%s", prod.head())
    logger.info("Raw Catalogue:\n%s", cat.head())

    for col in ["Seasonal"]:
        if col in prod.columns:
            prod[col] = to_bool01(prod[col])
    for col in ["SpringSummer", "DiscontinuedTF"]:
        if col in cat.columns:
            cat[col] = to_bool01(cat[col])

    if "DIorDOM" in prod.columns:
        prod["DIorDOM"] = prod["DIorDOM"].str.strip().str.upper()
        valid = {"DI", "DOM"}
        num_invalid = (~prod["DIorDOM"].isin(valid)).sum()
        if num_invalid:
            logger.warning("%d records have DIorDOM not in %s", num_invalid, valid)

    if "Status" in cat.columns:
        cat["Status"] = cat["Status"].str.strip().str.upper()
        valid_status = {"RI", "RO"}
        num_invalid = (~cat["Status"].isin(valid_status)).sum()
        if num_invalid:
            logger.warning("%d records have Status not in %s", num_invalid, valid_status)

    logger.info("Dtypes after normalization:\n%s\n%s", prod.dtypes, cat.dtypes)

    check_duplicates(prod, ["ProductKey"], "ProductDetails")
    check_duplicates(cat, ["CatEdition", "ProductKey", "WeeksOut"], "CatalogueDiscontinuation")
    missingness(prod, "ProductDetails")
    missingness(cat, "CatalogueDiscontinuation")

    df = cat.merge(prod, on="ProductKey", how="left", validate="many_to_one")
    logger.info("Joined df shape: %s", df.shape)
    coverage = df["ProductKey"].notna().mean() * 100
    logger.info("Join coverage (ProductDetails found): %.2f%%", coverage)

    df.to_parquet(os.path.join(output_dir, "eda_joined.parquet"), index=False)

    cols_to_keep = [c for c in [
        "CatEdition","SpringSummer","ProductKey","WeeksOut","Status",
        "SalePriceIncVAT","ForecastPerWeek","ActualsPerWeek","Fcast_to_Actual",
        "DiscontinuedTF","Supplier","HierarchyLevel1","HierarchyLevel2","DIorDOM","Seasonal"
    ] if c in df.columns]
    model_df = df[cols_to_keep].copy()
    model_df.to_parquet(os.path.join(output_dir, "eda_model_ready.parquet"), index=False)

    filtered_df = df[df["WeeksOut"].isin([-12, -8, -4])].copy()
    filtered_df.to_parquet(
        os.path.join(output_dir, "filtered_dataset.parquet"), index=False
    )

    # prepare for training
    y = filtered_df.pop("DiscontinuedTF").astype("int64")
    for col in ["DIorDOM", "Status"]:
        if col in filtered_df.columns:
            filtered_df[col] = filtered_df[col].astype("category").cat.codes
    dataset = pd.concat([y, filtered_df], axis=1).fillna(0)
    dataset = dataset.sample(frac=1, random_state=0)
    train, validation, test = np.split(
        dataset, [int(0.7 * len(dataset)), int(0.85 * len(dataset))]
    )

    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
