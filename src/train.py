"""
Model training with a strict **time-based** split.

Why not `train_test_split`?
---------------------------
The rows in this dataset are snapshots of customer behaviour taken at one
moment in time, and the label is "did this customer buy again in the *next*
90 days". A random split mixes customers observed at the end of 2011 into the
training set and customers observed in mid-2011 into the test set. Because
recency/monetary behaviour drifts over the year (a Christmas shopper looks very
different from a June shopper), a random split lets the model memorise
period-specific patterns that are already "in the future" of the test rows and
inflates every metric. The model would look excellent offline and fail in
production.

What we do instead
------------------
A **rolling-origin (out-of-time) backtest**:

    TRAIN  feature window: all transactions <= 2011-06-12
           label   window: 2011-06-12 -> 2011-09-10   (90 days)
    TEST   feature window: all transactions <= 2011-09-10
           label   window: 2011-09-10 -> 2011-12-09   (90 days)

The test period is strictly *after* the training period, so no test-period
information can reach the model. This mirrors how the model would actually be
deployed: fit on history, predict the next quarter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from . import data as D
from .features import FEATURE_COLUMNS, build_features, feature_frame

RANDOM_STATE = 42

#: End of the training feature window == start of the training label window.
TRAIN_CUT_DATE = D.CUT_DATE - pd.Timedelta(days=90)  # 2011-06-12
TRAIN_LABEL_END = D.CUT_DATE  # 2011-09-10


@dataclass
class Split:
    """One out-of-time evaluation window."""

    name: str
    cut_date: pd.Timestamp
    label_end: pd.Timestamp
    snapshot_date: pd.Timestamp
    features: pd.DataFrame
    labels: pd.DataFrame
    transactions: pd.DataFrame = field(repr=False, default=None)

    @property
    def merged(self) -> pd.DataFrame:
        return self.features.merge(self.labels, on="CustomerID", how="inner")

    def X(self) -> pd.DataFrame:
        return feature_frame(self.features)

    def y(self) -> np.ndarray:
        return self.labels["churned"].to_numpy(dtype=int)

    @property
    def churn_rate(self) -> float:
        return float(self.labels["churned"].mean())

    @property
    def n_customers(self) -> int:
        return int(len(self.labels))


def build_split(dataset: D.Dataset, cut_date: pd.Timestamp, label_end: pd.Timestamp, name: str) -> Split:
    """Build features + labels for a single cut date."""
    feature_window = dataset.clean[dataset.clean["InvoiceDate"].dt.normalize() <= cut_date]
    label_window = dataset.clean[
        (dataset.clean["InvoiceDate"].dt.normalize() > cut_date)
        & (dataset.clean["InvoiceDate"].dt.normalize() <= label_end)
    ]

    snapshot = cut_date + pd.Timedelta(days=1)
    features = build_features(feature_window, dataset.cancelled, cut_date, snapshot)
    labels = D.build_labels(features["CustomerID"], label_window)

    return Split(
        name=name,
        cut_date=cut_date,
        label_end=label_end,
        snapshot_date=snapshot,
        features=features,
        labels=labels,
        transactions=feature_window,
    )


def make_splits(dataset: D.Dataset, verbose: bool = True) -> tuple[Split, Split]:
    """Return ``(train, test)`` as a rolling-origin out-of-time split."""
    train = build_split(dataset, TRAIN_CUT_DATE, TRAIN_LABEL_END, "train")
    test = build_split(dataset, D.CUT_DATE, D.DATA_END_DATE, "test")

    if verbose:
        print("\n[train] TIME-BASED SPLIT (no random train_test_split)")
        for s in (train, test):
            print(
                f"[train] {s.name:<5} features<= {s.cut_date.date()}  "
                f"label=({s.cut_date.date()} -> {s.label_end.date()}]  "
                f"customers={s.n_customers:>5,}  churn={s.churn_rate:6.1%}"
            )
    return train, test


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def build_models() -> dict[str, Any]:
    """The three models, with class imbalance handled explicitly.

    * LogisticRegression  -> ``class_weight="balanced"``
    * RandomForest        -> ``class_weight="balanced_subsample"``
    * GradientBoosting    -> no ``class_weight`` parameter, so we pass
      balanced sample weights to ``fit`` instead (the equivalent mechanism).
    """
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        C=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=RANDOM_STATE,
        ),
    }


def fit_model(name: str, model: Any, X_train: pd.DataFrame, y_train: np.ndarray) -> Any:
    """Fit one model, applying balanced sample weights where supported."""
    if name == "GradientBoosting":
        weights = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)
    return model


def train_all(
    train: Split,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit every model on the training window."""
    X_train, y_train = train.X(), train.y()
    fitted: dict[str, Any] = {}

    for name, model in build_models().items():
        fit_model(name, model, X_train, y_train)
        fitted[name] = model
        if verbose:
            print(f"[train] fitted {name}")

    return fitted


def select_best(metrics: dict[str, dict[str, float]]) -> str:
    """Pick the best model by ROC-AUC, breaking ties on PR-AUC."""
    return max(metrics, key=lambda n: (metrics[n]["roc_auc"], metrics[n]["pr_auc"]))


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Positive-class probability, robust to pipelines."""
    return model.predict_proba(X)[:, 1]
