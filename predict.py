#!/usr/bin/env python
"""
Score customers with the trained churn model.

Examples
--------
Score the bundled raw-transaction example (6 customers):

    python predict.py --input data/sample/predict_example.csv

Score the full sample transaction file and write the scores out:

    python predict.py --input data/sample/online_retail_sample.csv \
                      --output reports/predictions.csv

Score a file that already contains the engineered features:

    python predict.py --input my_features.csv --threshold 0.4

The script auto-detects which of the two input shapes it received:

* **raw transactions** - the eight Online Retail columns
  (``InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice,
  CustomerID, Country``). Features are rebuilt with the same code the model
  was trained with.
* **engineered features** - the nine columns listed in
  ``src/features.py`` (``recency`` ... ``cancellation_ratio``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import CUT_DATE, SNAPSHOT_DATE, clean_transactions  # noqa: E402
from src.features import FEATURE_COLUMNS, build_features, feature_frame  # noqa: E402

DEFAULT_MODEL = REPO_ROOT / "models" / "churn_model.joblib"
RAW_REQUIRED = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}


def load_model(path: Path):
    """Return ``(estimator, metadata)`` from a joblib artifact."""
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload
    return payload, {}


def risk_tier(probability: float, low: float, high: float) -> str:
    if probability >= high:
        return "HIGH"
    if probability >= low:
        return "MEDIUM"
    return "LOW"


def features_from_transactions(
    df: pd.DataFrame,
    cut_date: pd.Timestamp,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Rebuild the training features from a raw transaction file."""
    clean, cancelled, _ = clean_transactions(df)

    in_window = clean[clean["InvoiceDate"].dt.normalize() <= cut_date]
    if in_window.empty:
        raise SystemExit(
            f"No transactions on or before the cut date {cut_date.date()}. "
            "Nothing can be scored."
        )

    features = build_features(in_window, cancelled, cut_date, snapshot_date)
    return features


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score customers for churn risk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="CSV of customers or raw transactions.")
    parser.add_argument("--model", "-m", default=str(DEFAULT_MODEL), help="Path to the joblib model.")
    parser.add_argument("--output", "-o", default=None, help="Optional path for a scored CSV.")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Decision threshold (default 0.5).")
    parser.add_argument(
        "--cut-date",
        default=None,
        help="Feature-window cut date (YYYY-MM-DD). Defaults to the value stored in the model.",
    )
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="'Today' used for recency. Defaults to cut date + 1 day.",
    )
    parser.add_argument("--top", type=int, default=25, help="How many rows to print (default 25).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}\nRun `python run_all.py` first.")

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    model, meta = load_model(model_path)
    stored_features = meta.get("feature_columns", FEATURE_COLUMNS)

    if args.cut_date:
        cut_date = pd.Timestamp(args.cut_date)
        # A new cut date implies a new snapshot unless one was given explicitly.
        snapshot_date = (
            pd.Timestamp(args.snapshot_date)
            if args.snapshot_date
            else cut_date + pd.Timedelta(days=1)
        )
    else:
        cut_date = pd.Timestamp(meta.get("cut_date", CUT_DATE.date().isoformat()))
        snapshot_date = pd.Timestamp(
            args.snapshot_date or meta.get("snapshot_date", SNAPSHOT_DATE.date().isoformat())
        )

    df = pd.read_csv(input_path)
    columns = set(df.columns)

    print("=" * 72)
    print("CHURN PREDICTION")
    print("=" * 72)
    print(f"model      : {meta.get('model_name', model_path.stem)}")
    print(f"artifact   : {model_path}")
    print(f"input      : {input_path}  ({len(df):,} rows)")
    print(f"cut date   : {cut_date.date()}   snapshot: {snapshot_date.date()}")
    print(f"threshold  : {args.threshold:.2f}")

    if set(stored_features).issubset(columns):
        mode = "engineered features"
        features = df.copy()
        if "CustomerID" not in features.columns:
            features["CustomerID"] = range(1, len(features) + 1)
    elif RAW_REQUIRED.issubset(columns):
        mode = "raw transactions"
        input_customers = set(df["CustomerID"].dropna().unique())
        features = features_from_transactions(df, cut_date, snapshot_date)
        dropped = input_customers - set(features["CustomerID"].unique())
        if dropped:
            print(
                f"note       : {len(dropped)} customer(s) had no transactions on or before "
                f"{cut_date.date()} and were skipped"
            )
    else:
        missing_raw = sorted(RAW_REQUIRED - columns)
        missing_feat = sorted(set(stored_features) - columns)
        raise SystemExit(
            "Could not recognise the input file.\n"
            f"  missing raw-transaction columns : {missing_raw}\n"
            f"  missing engineered features     : {missing_feat}"
        )

    print(f"input mode : {mode}")

    X = feature_frame(features)
    probabilities = model.predict_proba(X)[:, 1]

    scored = pd.DataFrame(
        {
            "CustomerID": features["CustomerID"].to_numpy(),
            "churn_probability": probabilities.round(4),
            "prediction": (probabilities >= args.threshold).astype(int),
        }
    )
    scored["prediction_label"] = scored["prediction"].map({1: "churn", 0: "retain"})
    scored["risk_tier"] = [
        risk_tier(p, 0.33, 0.66) for p in scored["churn_probability"]
    ]
    scored = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    print(f"customers  : {len(scored):,}")
    print(
        f"flagged    : {int(scored['prediction'].sum()):,} "
        f"({scored['prediction'].mean():.1%}) at threshold {args.threshold:.2f}"
    )
    print(f"mean prob  : {scored['churn_probability'].mean():.4f}")
    print("-" * 72)

    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(scored.head(args.top).to_string(index=False))

    if len(scored) > args.top:
        print(f"... {len(scored) - args.top:,} more rows")

    print("-" * 72)
    print("risk tier breakdown:")
    tiers = scored["risk_tier"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0).astype(int)
    for tier, count in tiers.items():
        print(f"  {tier:<7} {count:>6,}  ({count / len(scored):.1%})")
    print("=" * 72)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out_path, index=False)
        print(f"wrote {out_path}  ({out_path.stat().st_size:,} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
