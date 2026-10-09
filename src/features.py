"""
Feature engineering for the churn model.

All features are computed **only** from the feature window
(``InvoiceDate <= CUT_DATE``) and use a snapshot of ``CUT_DATE + 1 day`` as
"today". Nothing that happens after the cut date can reach the feature matrix,
which is what keeps the label window honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Order matters: this list defines the model's column order everywhere.
FEATURE_COLUMNS = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "total_items",
    "unique_skus",
    "customer_age_days",
    "cancellation_count",
    "cancellation_ratio",
]

FEATURE_DESCRIPTIONS = {
    "recency": "Days between the customer's last purchase and the snapshot date (cut + 1 day).",
    "frequency": "Number of distinct invoices in the feature window.",
    "monetary": "Total revenue (Quantity x UnitPrice) in the feature window.",
    "avg_order_value": "monetary / frequency.",
    "total_items": "Total units purchased in the feature window.",
    "unique_skus": "Number of distinct StockCodes purchased.",
    "customer_age_days": "Days between the customer's first purchase and the cut date.",
    "cancellation_count": "Distinct cancelled invoices (InvoiceNo starting with 'C') before the cut.",
    "cancellation_ratio": "cancellation_count / (cancellation_count + frequency).",
}


def build_features(
    feature_window: pd.DataFrame,
    cancelled: pd.DataFrame,
    cut_date: pd.Timestamp,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate transaction rows into one row of features per customer.

    Parameters
    ----------
    feature_window:
        Cleaned transactions with ``InvoiceDate <= cut_date``.
    cancelled:
        Cancellation rows kept aside during cleaning (see ``src.data``).
    cut_date:
        End of the feature window.
    snapshot_date:
        Date used as "today" for recency. Defaults to ``cut_date + 1 day``.
    """
    if snapshot_date is None:
        snapshot_date = cut_date + pd.Timedelta(days=1)

    if feature_window.empty:
        raise ValueError("Feature window is empty -- nothing to featurise.")

    df = feature_window.copy()
    df["_date"] = df["InvoiceDate"].dt.normalize()

    grouped = df.groupby("CustomerID", sort=True)
    features = pd.DataFrame(
        {
            "last_purchase": grouped["_date"].max(),
            "first_purchase": grouped["_date"].min(),
            "frequency": grouped["InvoiceNo"].nunique(),
            "monetary": grouped["revenue"].sum(),
            "total_items": grouped["Quantity"].sum(),
            "unique_skus": grouped["StockCode"].nunique(),
        }
    )

    snapshot = pd.Timestamp(snapshot_date).normalize()
    features["recency"] = (snapshot - features["last_purchase"]).dt.days
    features["customer_age_days"] = (pd.Timestamp(cut_date) - features["first_purchase"]).dt.days
    features["avg_order_value"] = features["monetary"] / features["frequency"]

    # -- cancellation behaviour (from the rows cleaning rule 2 removed) ------ #
    if cancelled is not None and not cancelled.empty:
        cancel = cancelled.copy()
        cancel["_date"] = cancel["InvoiceDate"].dt.normalize()
        cancel = cancel[cancel["_date"] <= pd.Timestamp(cut_date)]
        counts = cancel.groupby("CustomerID")["InvoiceNo"].nunique()
    else:
        counts = pd.Series(dtype="int64")

    features["cancellation_count"] = counts.reindex(features.index).fillna(0).astype("int64")
    features["cancellation_ratio"] = features["cancellation_count"] / (
        features["cancellation_count"] + features["frequency"]
    )

    features = features.reset_index()
    features["monetary"] = features["monetary"].round(2)
    features["avg_order_value"] = features["avg_order_value"].round(2)

    return features


def recency_bucket(recency: pd.Series) -> pd.Categorical:
    """Bucket customers by recency, matching the buckets used in the report."""
    bins = [-1, 30, 90, 180, 400, np.inf]
    labels = ["0-30d", "31-90d", "91-180d", "181-400d", "400d+"]
    return pd.cut(recency, bins=bins, labels=labels)


def feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Select + order the modelling columns."""
    return features[FEATURE_COLUMNS].astype("float64")


def dataset_summary(features: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Small dict of headline facts used by the report."""
    merged = features.merge(labels, on="CustomerID", how="inner")
    churn_rate = float(merged["churned"].mean())
    return {
        "n_customers": int(len(merged)),
        "n_churned": int(merged["churned"].sum()),
        "n_retained": int((1 - merged["churned"]).sum()),
        "churn_rate": churn_rate,
        "majority_baseline_accuracy": float(max(churn_rate, 1 - churn_rate)),
        "median_recency": float(merged["recency"].median()),
        "median_monetary": float(merged["monetary"].median()),
    }
