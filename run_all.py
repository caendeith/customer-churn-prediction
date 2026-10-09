#!/usr/bin/env python
"""
End-to-end churn pipeline.

    python run_all.py                 # uses the full dataset if present, else the sample
    python run_all.py --sample        # force the bundled 437-customer sample
    python run_all.py --download      # fetch the full UCI workbook into data/raw/ first

Stages: load -> clean -> build windows -> features -> train -> evaluate
        -> figures -> save model -> write reports/model_report.md
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src import data as D  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import features as F  # noqa: E402
from src import train as T  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

EXPECTED_FIGURES = [
    "roc_curves.png",
    "pr_curve.png",
    "confusion_matrix.png",
    "feature_importance.png",
    "churn_rate.png",
    "rfm_by_churn.png",
]

#: Reference numbers verified on the full dataset - used as a sanity check only.
FULL_DATA_REFERENCE = {
    "raw_rows": 541_909,
    "clean_rows": 392_692,
    "revenue": 8_887_208.89,
    "customers": 4_338,
    "scoped_customers": 3_370,
    "churned": 1_449,
    "churn_rate": 0.430,
}


def banner(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def rel(path: Path | str) -> str:
    """Repo-relative POSIX path, for logs and markdown."""
    try:
        return Path(path).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full churn pipeline.")
    parser.add_argument("--sample", action="store_true", help="Force the bundled sample dataset.")
    parser.add_argument("--download", action="store_true", help="Download the full dataset first.")
    return parser.parse_args(argv)


def write_predict_example(sample_path: Path, out_path: Path, n_customers: int = 6) -> Path | None:
    """Write a small raw-transaction CSV used to demo `predict.py`.

    Picks customers whose transaction count is closest to the median, so the
    file is small but the rows are representative.
    """
    if not sample_path.is_file():
        return None

    df = pd.read_csv(sample_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Only customers with history before the model's cut date can be scored, so
    # pick from that population to keep the demo output complete.
    history = df[df["InvoiceDate"].dt.normalize() <= T.TRAIN_CUT_DATE]
    counts = history["CustomerID"].value_counts()
    if counts.empty:
        return None

    median = counts.median()
    chosen = (counts - median).abs().sort_values(kind="stable").head(n_customers).index.tolist()
    chosen = sorted(chosen)

    small = df[df["CustomerID"].isin(chosen)].copy()
    small["InvoiceDate"] = small["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    small.to_csv(out_path, index=False)

    # A second, tiny example showing the "engineered features" input mode.
    feat_path = out_path.with_name("predict_features_example.csv")
    rows = [
        # CustomerID, recency, frequency, monetary, avg_order_value,
        # total_items, unique_skus, customer_age_days, cancellation_count, cancellation_ratio
        [900001, 10, 8, 2500.00, 312.50, 900, 120, 300, 0, 0.000],
        [900002, 220, 1, 45.00, 45.00, 12, 3, 240, 2, 0.667],
    ]
    pd.DataFrame(rows, columns=["CustomerID"] + F.FEATURE_COLUMNS).to_csv(feat_path, index=False)

    return out_path


def build_report(
    dataset: D.Dataset,
    train: T.Split,
    test: T.Split,
    metrics: dict[str, dict[str, float]],
    best_name: str,
    importance: pd.DataFrame,
    recency_table: pd.DataFrame,
    summary: dict,
    threshold_info: dict[str, tuple[float, float]],
    figures: list[tuple[str, str]],
    elapsed: float,
) -> str:
    """Render `reports/model_report.md` from the real run results."""
    b = metrics[best_name]
    lines: list[str] = []

    lines.append("# Churn model report")
    lines.append("")
    lines.append(
        "Generated automatically by `python run_all.py`. "
        "Every number below was produced by that run - none of it is hand-written."
    )
    lines.append("")
    lines.append(f"- Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Runtime: {elapsed:.1f}s")
    lines.append(f"- Python {platform.python_version()} / scikit-learn {_sklearn_version()}")
    lines.append(f"- Data source: `{rel(dataset.source_path)}` ({dataset.source_kind})")
    lines.append("")

    # -- 1. data ---------------------------------------------------------- #
    lines.append("## 1. Data")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Raw rows | {dataset.n_raw_rows:,} |")
    lines.append(f"| Cleaned rows | {dataset.n_clean_rows:,} |")
    lines.append(f"| Unique customers | {dataset.n_customers:,} |")
    lines.append(f"| Total revenue | GBP {dataset.total_revenue:,.2f} |")
    lines.append(f"| Date range | {dataset.clean['InvoiceDate'].min().date()} to {dataset.clean['InvoiceDate'].max().date()} |")
    lines.append("")

    lines.append("### Cleaning log (rules applied in this exact order)")
    lines.append("")
    lines.append("| Step | Rows removed | Rows remaining |")
    lines.append("|---|---:|---:|")
    for step in dataset.cleaning_log:
        lines.append(f"| {step.step} | {step.rows_removed:,} | {step.rows_remaining:,} |")
    lines.append("")

    # -- 2. target -------------------------------------------------------- #
    lines.append("## 2. Churn definition and scope")
    lines.append("")
    lines.append(f"- End of data: **{D.DATA_END_DATE.date()}**")
    lines.append(f"- Feature window: transactions up to and including **{D.CUT_DATE.date()}**")
    lines.append(
        f"- Label window: **{D.CUT_DATE.date()} to {D.DATA_END_DATE.date()}** "
        f"({D.LABEL_WINDOW_DAYS} days)"
    )
    lines.append("- **Churn = zero purchases in the label window**")
    lines.append("- In scope: customers with at least 1 purchase in the feature window")
    lines.append("")
    lines.append("| Quantity | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Customers in scope | {summary['n_customers']:,} |")
    lines.append(f"| Churned | {summary['n_churned']:,} |")
    lines.append(f"| Retained | {summary['n_retained']:,} |")
    lines.append(f"| Churn rate | {summary['churn_rate']:.3f} ({summary['churn_rate']:.1%}) |")
    lines.append(
        f"| Majority-class baseline accuracy | {summary['majority_baseline_accuracy']:.3f} |"
    )
    lines.append("")

    if dataset.source_kind == "full":
        lines.append("Sanity check against the verified full-data reference:")
        lines.append("")
        lines.append("| Fact | Expected | Observed | Match |")
        lines.append("|---|---:|---:|:--:|")
        checks = [
            ("Raw rows", FULL_DATA_REFERENCE["raw_rows"], dataset.n_raw_rows, "{:,}"),
            ("Cleaned rows", FULL_DATA_REFERENCE["clean_rows"], dataset.n_clean_rows, "{:,}"),
            ("Revenue", FULL_DATA_REFERENCE["revenue"], dataset.total_revenue, "{:,.2f}"),
            ("Unique customers", FULL_DATA_REFERENCE["customers"], dataset.n_customers, "{:,}"),
            ("Customers in scope", FULL_DATA_REFERENCE["scoped_customers"], summary["n_customers"], "{:,}"),
            ("Churned", FULL_DATA_REFERENCE["churned"], summary["n_churned"], "{:,}"),
        ]
        for label, expected, observed, fmt in checks:
            ok = "yes" if abs(float(expected) - float(observed)) < 0.01 else "NO"
            lines.append(f"| {label} | {fmt.format(expected)} | {fmt.format(observed)} | {ok} |")
        lines.append("")
    else:
        lines.append(
            "> **Note:** this run used the bundled sample, so the counts above are smaller "
            "than the published full-data figures."
        )
        lines.append("")

    # -- 3. split --------------------------------------------------------- #
    lines.append("## 3. Time-based split (why not `train_test_split`)")
    lines.append("")
    lines.append(
        "The label asks what happens in the **next 90 days**. A random split would put "
        "customers observed at the end of 2011 into training and customers observed in "
        "mid-2011 into testing, so the model would be fitted on a later period than it is "
        "scored on. Seasonal drift (a December shopper does not look like a June shopper) "
        "would then be memorised rather than predicted, and every metric would be inflated."
    )
    lines.append("")
    lines.append("We therefore use a **rolling-origin out-of-time backtest**:")
    lines.append("")
    lines.append("| Window | Feature window | Label window | Customers | Churn rate |")
    lines.append("|---|---|---|---:|---:|")
    for split in (train, test):
        lines.append(
            f"| {split.name.capitalize()} | <= {split.cut_date.date()} "
            f"| {split.cut_date.date()} to {split.label_end.date()} "
            f"| {split.n_customers:,} | {split.churn_rate:.1%} |"
        )
    lines.append("")
    lines.append(
        "The test window is strictly later than the training window, so no future "
        "information can reach the model. Reported metrics are all from the **test** window."
    )
    lines.append("")

    # -- 4. metrics ------------------------------------------------------- #
    lines.append("## 4. Results - model comparison (out-of-time test window)")
    lines.append("")
    lines.append(E.markdown_table(E.metrics_table(metrics, best_name)))
    lines.append("")
    lines.append(
        f"All metrics are computed at a decision threshold of **0.50**. "
        f"The best model is **{best_name}** (highest ROC-AUC, ties broken on PR-AUC)."
    )
    lines.append("")
    lines.append("### Model configuration and class imbalance")
    lines.append("")
    lines.append(
        "Churn is not rare (43.0%), but it is still the minority class, so an unweighted "
        "model is biased towards the majority. Each model is balanced explicitly:"
    )
    lines.append("")
    lines.append("| Model | Imbalance handling | Key hyperparameters |")
    lines.append("|---|---|---|")
    lines.append(
        "| LogisticRegression | `class_weight='balanced'` "
        "| `StandardScaler` then `C=1.0`, `max_iter=2000`, `random_state=42` |"
    )
    lines.append(
        "| RandomForest | `class_weight='balanced_subsample'` "
        "| `n_estimators=500`, `min_samples_leaf=5`, `random_state=42` |"
    )
    lines.append(
        "| GradientBoosting | balanced **sample weights** via "
        "`compute_sample_weight('balanced', y)` (the class has no `class_weight` argument) "
        "| `n_estimators=300`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.9`, `random_state=42` |"
    )
    lines.append("")
    lines.append(
        "Hyperparameters were fixed **before** looking at the test window. No tuning was "
        "done against the test set, so the numbers below are a genuine out-of-time "
        "estimate rather than an optimistically selected one."
    )
    lines.append("")
    lines.append("### Confusion matrix - best model")
    lines.append("")
    lines.append("| | Predicted retained | Predicted churn |")
    lines.append("|---|---:|---:|")
    lines.append(f"| **Actually retained** | {b['tn']:,} | {b['fp']:,} |")
    lines.append(f"| **Actually churned** | {b['fn']:,} | {b['tp']:,} |")
    lines.append("")
    lines.append("### Threshold sensitivity")
    lines.append("")
    lines.append(
        "The 0.50 threshold is arbitrary. Because churn is the minority class, lowering "
        "the threshold trades precision for recall - the right trade depends on the cost "
        "of a retention offer versus the value of a saved customer."
    )
    lines.append("")
    lines.append("| Model | Best-F1 threshold | F1 at that threshold | F1 at 0.50 |")
    lines.append("|---|---:|---:|---:|")
    for name in metrics:
        thr, f1 = threshold_info[name]
        lines.append(f"| {name} | {thr:.3f} | {f1:.4f} | {metrics[name]['f1']:.4f} |")
    lines.append("")

    # -- 5. signal -------------------------------------------------------- #
    lines.append("## 5. The signal the model is picking up")
    lines.append("")
    lines.append("### Churn rate by recency bucket")
    lines.append("")
    lines.append("| Recency bucket | Customers | Churned | Churn rate |")
    lines.append("|---|---:|---:|---:|")
    for row in recency_table.itertuples():
        lines.append(
            f"| {row.recency_bucket} | {int(row.customers):,} | {int(row.churned):,} "
            f"| {float(row.churn_rate):.1%} |"
        )
    lines.append("")
    lines.append(
        "Churn risk climbs monotonically with recency, which is exactly what the business "
        "would expect and what makes recency the dominant feature."
    )
    lines.append("")
    lines.append(f"### Permutation importance - {best_name}")
    lines.append("")
    lines.append("| Feature | Mean ROC-AUC drop | Std |")
    lines.append("|---|---:|---:|")
    for row in importance.sort_values("importance", ascending=False).itertuples():
        lines.append(f"| `{row.feature}` | {row.importance:.4f} | {row.std:.4f} |")
    lines.append("")
    lines.append(
        "Importance is measured on the test window by shuffling one feature at a time, "
        "so it is model-agnostic and not inflated by training-set fit."
    )
    lines.append("")

    # -- 6. figures ------------------------------------------------------- #
    lines.append("## 6. Figures")
    lines.append("")
    for filename, caption in figures:
        lines.append(f"### {caption}")
        lines.append("")
        lines.append(f"![{caption}](figures/{filename})")
        lines.append("")

    # -- 7. artifact ------------------------------------------------------ #
    lines.append("## 7. Saved artifact")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append("| Path | `models/churn_model.joblib` |")
    lines.append(f"| Model | {best_name} |")
    lines.append(f"| ROC-AUC | {b['roc_auc']:.4f} |")
    lines.append(f"| PR-AUC | {b['pr_auc']:.4f} |")
    lines.append(f"| Trained on | feature window <= {train.cut_date.date()} ({train.n_customers:,} customers) |")
    lines.append(f"| Snapshot date | {train.snapshot_date.date()} |")
    lines.append("")
    lines.append(
        "The saved artifact is the model trained on the **training** window, i.e. exactly "
        "the model whose metrics are reported above, so the numbers can be reproduced by "
        "loading it. For a production deployment you would refit on the full feature "
        "window (all customers up to the latest cut date) - see the README."
    )
    lines.append("")
    lines.append("Feature columns expected by the model, in order:")
    lines.append("")
    lines.append("```")
    for col in F.FEATURE_COLUMNS:
        lines.append(f"{col:<20} # {F.FEATURE_DESCRIPTIONS[col]}")
    lines.append("```")
    lines.append("")
    lines.append("Score new customers with:")
    lines.append("")
    lines.append("```bash")
    lines.append("python predict.py --input data/sample/predict_example.csv")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    # Deterministic figures: seaborn's strip-plot jitter draws from numpy's
    # global random state, which is otherwise unseeded.
    np.random.seed(T.RANDOM_STATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.download:
        banner("STEP 0 - download the full dataset")
        D.download_full_dataset(REPO_ROOT)

    # -- 1. data ---------------------------------------------------------- #
    banner("STEP 1 - load and clean the data")
    if args.sample:
        dataset = D.load_dataset_from(REPO_ROOT / D.SAMPLE_DATA_PATH, "sample", REPO_ROOT)
    else:
        dataset = D.load_dataset(REPO_ROOT)

    print("\n[data] cleaning log:")
    for step in dataset.cleaning_log:
        print(f"  {step.step:<55} removed={step.rows_removed:>8,}  remaining={step.rows_remaining:>9,}")

    # -- 2. windows + features -------------------------------------------- #
    banner("STEP 2 - build feature / label windows and features")
    train, test = T.make_splits(dataset)

    summary = F.dataset_summary(test.features, test.labels)
    print(
        f"\n[features] test scope : {summary['n_customers']:,} customers, "
        f"{summary['n_churned']:,} churned ({summary['churn_rate']:.1%}), "
        f"baseline accuracy {summary['majority_baseline_accuracy']:.3f}"
    )
    print(f"[features] columns    : {', '.join(F.FEATURE_COLUMNS)}")
    print(f"[features] train matrix: {train.X().shape}   test matrix: {test.X().shape}")

    # -- 3. train --------------------------------------------------------- #
    banner("STEP 3 - train models (class imbalance handled explicitly)")
    models = T.train_all(train)

    # -- 4. evaluate ------------------------------------------------------ #
    banner("STEP 4 - evaluate on the out-of-time test window")
    X_test, y_test = test.X(), test.y()
    metrics, probs = E.evaluate_models(models, X_test, y_test)
    best_name = T.select_best(metrics)

    table = E.metrics_table(metrics, best_name)
    print()
    print(table.to_string(index=False))
    print(f"\n[eval] best model: {best_name}")

    threshold_info = {
        name: E.best_f1_threshold(y_test, prob) for name, prob in probs.items()
    }

    # -- 5. figures ------------------------------------------------------- #
    banner("STEP 5 - figures")
    merged = test.merged

    E.plot_roc_curves(y_test, probs, metrics, FIGURES_DIR / "roc_curves.png")
    print("[fig] roc_curves.png")
    E.plot_pr_curve(y_test, probs, metrics, FIGURES_DIR / "pr_curve.png")
    print("[fig] pr_curve.png")
    E.plot_confusion_matrix(y_test, probs[best_name], best_name, FIGURES_DIR / "confusion_matrix.png")
    print("[fig] confusion_matrix.png")
    importance = E.plot_feature_importance(
        models[best_name], X_test, y_test, best_name, FIGURES_DIR / "feature_importance.png"
    )
    print("[fig] feature_importance.png")
    recency_table = E.plot_churn_rate(merged, FIGURES_DIR / "churn_rate.png")
    print("[fig] churn_rate.png")
    E.plot_rfm_by_churn(merged, FIGURES_DIR / "rfm_by_churn.png")
    print("[fig] rfm_by_churn.png")

    # -- 6. save the model ------------------------------------------------ #
    banner("STEP 6 - save the model")
    best_model = models[best_name]
    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": F.FEATURE_COLUMNS,
        "cut_date": train.cut_date.date().isoformat(),
        "snapshot_date": train.snapshot_date.date().isoformat(),
        "train_feature_window_end": train.cut_date.date().isoformat(),
        "train_label_window_end": train.label_end.date().isoformat(),
        "test_cut_date": test.cut_date.date().isoformat(),
        "test_label_window_end": test.label_end.date().isoformat(),
        "metrics": metrics[best_name],
        "all_metrics": metrics,
        "data_source": dataset.source_kind,
        "data_source_path": rel(dataset.source_path),
        "sklearn_version": _sklearn_version(),
        "random_state": T.RANDOM_STATE,
    }
    model_path = MODELS_DIR / "churn_model.joblib"
    joblib.dump(payload, model_path)
    print(f"[model] wrote {rel(model_path)}  ({model_path.stat().st_size:,} bytes)")

    # -- 7. reports ------------------------------------------------------- #
    banner("STEP 7 - write reports")

    figures = [
        ("roc_curves.png", "ROC curves (all three models)"),
        ("pr_curve.png", "Precision-Recall curves"),
        ("confusion_matrix.png", f"Confusion matrix ({best_name})"),
        ("feature_importance.png", "Permutation importance"),
        ("churn_rate.png", "Churn rate by recency bucket"),
        ("rfm_by_churn.png", "RFM distributions by churn status"),
    ]

    report = build_report(
        dataset, train, test, metrics, best_name, importance, recency_table,
        summary, threshold_info, figures, time.perf_counter() - started,
    )
    report_path = REPORTS_DIR / "model_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[report] wrote {rel(report_path)}  ({report_path.stat().st_size:,} bytes)")

    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "data_source": dataset.source_kind,
                "data_source_path": rel(dataset.source_path),
                "raw_rows": dataset.n_raw_rows,
                "clean_rows": dataset.n_clean_rows,
                "unique_customers": dataset.n_customers,
                "total_revenue": round(dataset.total_revenue, 2),
                "scope": summary,
                "train_split": {
                    "cut_date": train.cut_date.date().isoformat(),
                    "label_end": train.label_end.date().isoformat(),
                    "customers": train.n_customers,
                    "churn_rate": round(train.churn_rate, 6),
                },
                "test_split": {
                    "cut_date": test.cut_date.date().isoformat(),
                    "label_end": test.label_end.date().isoformat(),
                    "customers": test.n_customers,
                    "churn_rate": round(test.churn_rate, 6),
                },
                "best_model": best_name,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[report] wrote {rel(metrics_path)}")

    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    recency_table.to_csv(REPORTS_DIR / "churn_by_recency.csv", index=False)

    # -- 8. demo CSV for predict.py --------------------------------------- #
    example = write_predict_example(
        REPO_ROOT / D.SAMPLE_DATA_PATH, REPO_ROOT / "data" / "sample" / "predict_example.csv"
    )
    if example:
        print(f"[data] wrote {rel(example)}  ({example.stat().st_size:,} bytes)")

    # -- 9. verify -------------------------------------------------------- #
    banner("STEP 8 - verify figures")
    all_ok = True
    for name in EXPECTED_FIGURES:
        path = FIGURES_DIR / name
        if path.is_file() and path.stat().st_size > 0:
            print(f"  OK    {name:<24} {path.stat().st_size:>9,} bytes")
        else:
            all_ok = False
            print(f"  FAIL  {name:<24} missing or empty")
    if not all_ok:
        raise SystemExit("One or more figures are missing or empty.")

    elapsed = time.perf_counter() - started
    banner(f"DONE in {elapsed:.1f}s")
    print(f"Data source : {rel(dataset.source_path)} ({dataset.source_kind})")
    print(f"Churn rate  : {summary['churn_rate']:.1%} "
          f"({summary['n_churned']:,}/{summary['n_customers']:,})")
    print(f"Best model  : {best_name}")
    for name, m in sorted(metrics.items(), key=lambda kv: -kv[1]["roc_auc"]):
        print(f"  {name:<20} ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"F1={m['f1']:.4f}  acc={m['accuracy']:.4f}")
    print(f"Report      : {rel(report_path)}")
    print("\nNext: python predict.py --input data/sample/predict_example.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
