"""Metrics and figures for the churn model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must happen before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .features import FEATURE_COLUMNS, recency_bucket

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 11})

PALETTE = {
    "LogisticRegression": "#4C72B0",
    "RandomForest": "#55A868",
    "GradientBoosting": "#C44E52",
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Headline classification metrics at a fixed decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Threshold that maximises F1 on the given predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    idx = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[idx]) if len(thresholds) else 0.5, float(f1[idx])


def evaluate_models(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Score every model on the held-out test window."""
    metrics: dict[str, dict[str, float]] = {}
    probs: dict[str, np.ndarray] = {}

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        probs[name] = y_prob
        metrics[name] = compute_metrics(y_test, y_prob)

    return metrics, probs


def metrics_table(metrics: dict[str, dict[str, float]], best: str) -> pd.DataFrame:
    """Markdown-friendly comparison table, sorted by ROC-AUC."""
    rows = []
    for name, m in metrics.items():
        rows.append(
            {
                "Model": f"**{name}**" if name == best else name,
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "PR-AUC": f"{m['pr_auc']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1": f"{m['f1']:.4f}",
                "Accuracy": f"{m['accuracy']:.4f}",
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )


def markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_roc_curves(
    y_test: np.ndarray,
    probs: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for name, y_prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {metrics[name]['roc_auc']:.3f})",
            color=PALETTE.get(name),
            lw=2.2,
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - out-of-time test window\n(2011-09-10 -> 2011-12-09)")
    ax.legend(loc="lower right", fontsize=11)
    _save(fig, out)


def plot_pr_curve(
    y_test: np.ndarray,
    probs: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    out: Path,
) -> None:
    prevalence = float(np.mean(y_test))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for name, y_prob in probs.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(
            recall,
            precision,
            label=f"{name} (AP = {metrics[name]['pr_auc']:.3f})",
            color=PALETTE.get(name),
            lw=2.2,
        )
    ax.axhline(
        prevalence,
        color="k",
        ls="--",
        lw=1.2,
        label=f"No-skill baseline (prevalence = {prevalence:.3f})",
    )
    ax.set_xlabel("Recall (churners found)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves - out-of-time test window")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=10)
    _save(fig, out)


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    out: Path,
    threshold: float = 0.5,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    pct = cm / cm.sum() * 100

    annot = np.array(
        [[f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)]
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted retained", "Predicted churn"],
        yticklabels=["Actually retained", "Actually churned"],
        annot_kws={"size": 13},
        ax=ax,
    )
    ax.set_title(f"Confusion matrix - {model_name}\n(threshold = {threshold:.2f})")
    _save(fig, out)


def plot_feature_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_name: str,
    out: Path,
) -> pd.DataFrame:
    """Model-agnostic permutation importance measured on the test window."""
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )
    imp = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(
        imp["feature"],
        imp["importance"],
        xerr=imp["std"],
        color="#4C72B0",
        error_kw={"ecolor": "#333333", "lw": 1.0, "capsize": 3},
    )
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title(f"Permutation importance - {model_name}\n(measured on the out-of-time test window)")
    _save(fig, out)
    return imp


def plot_churn_rate(merged: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Churn rate by recency bucket - the core business signal."""
    df = merged.copy()
    df["recency_bucket"] = recency_bucket(df["recency"])
    table = (
        df.groupby("recency_bucket", observed=True)["churned"]
        .agg(customers="size", churned="sum", churn_rate="mean")
        .reset_index()
    )
    table["churn_rate"] = table["churn_rate"].astype(float)

    overall = float(df["churned"].mean())
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(
        table["recency_bucket"].astype(str),
        table["churn_rate"],
        color=sns.color_palette("rocket_r", n_colors=len(table)),
    )
    ax.axhline(overall, color="#333333", ls="--", lw=1.4, label=f"Overall churn rate = {overall:.1%}")
    for bar, row in zip(bars, table.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{row.churn_rate:.1%}\n(n={row.customers:,})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, min(1.0, table["churn_rate"].max() + 0.16))
    ax.set_xlabel("Recency bucket (days since last purchase at the cut date)")
    ax.set_ylabel("Churn rate")
    ax.set_title("Churn rate rises monotonically with recency")
    ax.legend(loc="upper left", fontsize=10)
    _save(fig, out)
    return table


def plot_rfm_by_churn(merged: pd.DataFrame, out: Path) -> None:
    """How the three RFM dimensions separate churners from retained customers."""
    df = merged.copy()
    df["Status"] = np.where(df["churned"] == 1, "Churned", "Retained")
    order = ["Retained", "Churned"]
    colors = {"Retained": "#55A868", "Churned": "#C44E52"}

    specs = [
        ("recency", "Recency (days)"),
        ("frequency", "Frequency (invoices)"),
        ("monetary", "Monetary (GBP)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (col, label) in zip(axes, specs):
        sns.boxplot(
            data=df,
            x="Status",
            y=col,
            order=order,
            hue="Status",
            palette=colors,
            legend=False,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            data=df.sample(min(len(df), 700), random_state=42),
            x="Status",
            y=col,
            order=order,
            color="black",
            alpha=0.25,
            size=2,
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("")
        if col in {"frequency", "monetary"}:
            ax.set_yscale("log")
            ax.set_ylabel(f"{label} (log scale)")
        else:
            ax.set_ylabel(label)

    fig.suptitle("RFM distributions by churn status (feature window)", y=1.02)
    _save(fig, out)


def plot_target_balance(train: Any, test: Any, out: Path) -> None:
    """Optional extra: class balance in each window."""
    fig, ax = plt.subplots(figsize=(6, 4.6))
    labels = ["Train\n(2011-06-12)", "Test\n(2011-09-10)"]
    retained = [(1 - train.churn_rate) * 100, (1 - test.churn_rate) * 100]
    churned = [train.churn_rate * 100, test.churn_rate * 100]
    x = np.arange(2)
    ax.bar(x, retained, 0.55, label="Retained", color="#55A868")
    ax.bar(x, churned, 0.55, bottom=retained, label="Churned", color="#C44E52")
    for i, (r, c) in enumerate(zip(retained, churned)):
        ax.text(i, r / 2, f"{r:.1f}%", ha="center", va="center", color="white", fontsize=11)
        ax.text(i, r + c / 2, f"{c:.1f}%", ha="center", va="center", color="white", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of customers (%)")
    ax.set_title("Class balance is similar in both windows")
    ax.legend(loc="lower right", fontsize=10)
    _save(fig, out)
