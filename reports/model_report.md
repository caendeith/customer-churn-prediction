# Churn model report

Generated automatically by `python run_all.py`. Every number below was produced by that run - none of it is hand-written.

- Generated: 2026-10-10 02:53:47
- Runtime: 41.8s
- Python 3.11.1 / scikit-learn 1.9.1
- Data source: `data/raw/Online Retail.xlsx` (full)

## 1. Data

| Metric | Value |
|---|---|
| Raw rows | 541,909 |
| Cleaned rows | 392,692 |
| Unique customers | 4,338 |
| Total revenue | GBP 8,887,208.89 |
| Date range | 2010-12-01 to 2011-12-09 |

### Cleaning log (rules applied in this exact order)

| Step | Rows removed | Rows remaining |
|---|---:|---:|
| 1. drop rows with missing CustomerID | 135,080 | 406,829 |
| 2. drop cancellations (InvoiceNo starts with 'C') | 8,905 | 397,924 |
| 3. drop Quantity <= 0 | 0 | 397,924 |
| 4. drop UnitPrice <= 0 | 40 | 397,884 |
| 5. drop exact duplicate rows | 5,192 | 392,692 |
| 6. revenue = Quantity * UnitPrice | 0 | 392,692 |

## 2. Churn definition and scope

- End of data: **2011-12-09**
- Feature window: transactions up to and including **2011-09-10**
- Label window: **2011-09-10 to 2011-12-09** (90 days)
- **Churn = zero purchases in the label window**
- In scope: customers with at least 1 purchase in the feature window

| Quantity | Value |
|---|---:|
| Customers in scope | 3,370 |
| Churned | 1,449 |
| Retained | 1,921 |
| Churn rate | 0.430 (43.0%) |
| Majority-class baseline accuracy | 0.570 |

Sanity check against the verified full-data reference:

| Fact | Expected | Observed | Match |
|---|---:|---:|:--:|
| Raw rows | 541,909 | 541,909 | yes |
| Cleaned rows | 392,692 | 392,692 | yes |
| Revenue | 8,887,208.89 | 8,887,208.89 | yes |
| Unique customers | 4,338 | 4,338 | yes |
| Customers in scope | 3,370 | 3,370 | yes |
| Churned | 1,449 | 1,449 | yes |

## 3. Time-based split (why not `train_test_split`)

The label asks what happens in the **next 90 days**. A random split would put customers observed at the end of 2011 into training and customers observed in mid-2011 into testing, so the model would be fitted on a later period than it is scored on. Seasonal drift (a December shopper does not look like a June shopper) would then be memorised rather than predicted, and every metric would be inflated.

We therefore use a **rolling-origin out-of-time backtest**:

| Window | Feature window | Label window | Customers | Churn rate |
|---|---|---|---:|---:|
| Train | <= 2011-06-12 | 2011-06-12 to 2011-09-10 | 2,821 | 52.3% |
| Test | <= 2011-09-10 | 2011-09-10 to 2011-12-09 | 3,370 | 43.0% |

The test window is strictly later than the training window, so no future information can reach the model. Reported metrics are all from the **test** window.

## 4. Results - model comparison (out-of-time test window)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|
| **LogisticRegression** | 0.7443 | 0.6342 | 0.5902 | 0.7585 | 0.6638 | 0.6697 |
| RandomForest | 0.7311 | 0.6070 | 0.5760 | 0.7764 | 0.6614 | 0.6582 |
| GradientBoosting | 0.7286 | 0.6043 | 0.5768 | 0.7778 | 0.6624 | 0.6591 |

All metrics are computed at a decision threshold of **0.50**. The best model is **LogisticRegression** (highest ROC-AUC, ties broken on PR-AUC).

### Model configuration and class imbalance

Churn is not rare (43.0%), but it is still the minority class, so an unweighted model is biased towards the majority. Each model is balanced explicitly:

| Model | Imbalance handling | Key hyperparameters |
|---|---|---|
| LogisticRegression | `class_weight='balanced'` | `StandardScaler` then `C=1.0`, `max_iter=2000`, `random_state=42` |
| RandomForest | `class_weight='balanced_subsample'` | `n_estimators=500`, `min_samples_leaf=5`, `random_state=42` |
| GradientBoosting | balanced **sample weights** via `compute_sample_weight('balanced', y)` (the class has no `class_weight` argument) | `n_estimators=300`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.9`, `random_state=42` |

Hyperparameters were fixed **before** looking at the test window. No tuning was done against the test set, so the numbers below are a genuine out-of-time estimate rather than an optimistically selected one.

### Confusion matrix - best model

| | Predicted retained | Predicted churn |
|---|---:|---:|
| **Actually retained** | 1,158 | 763 |
| **Actually churned** | 350 | 1,099 |

### Threshold sensitivity

The 0.50 threshold is arbitrary. Because churn is the minority class, lowering the threshold trades precision for recall - the right trade depends on the cost of a retention offer versus the value of a saved customer.

| Model | Best-F1 threshold | F1 at that threshold | F1 at 0.50 |
|---|---:|---:|---:|
| LogisticRegression | 0.393 | 0.6800 | 0.6638 |
| RandomForest | 0.393 | 0.6751 | 0.6614 |
| GradientBoosting | 0.360 | 0.6753 | 0.6624 |

## 5. The signal the model is picking up

### Churn rate by recency bucket

| Recency bucket | Customers | Churned | Churn rate |
|---|---:|---:|---:|
| 0-30d | 904 | 229 | 25.3% |
| 31-90d | 992 | 359 | 36.2% |
| 91-180d | 888 | 474 | 53.4% |
| 181-400d | 586 | 387 | 66.0% |

Churn risk climbs monotonically with recency, which is exactly what the business would expect and what makes recency the dominant feature.

### Permutation importance - LogisticRegression

| Feature | Mean ROC-AUC drop | Std |
|---|---:|---:|
| `recency` | 0.0446 | 0.0056 |
| `frequency` | 0.0436 | 0.0058 |
| `unique_skus` | 0.0193 | 0.0023 |
| `total_items` | 0.0157 | 0.0027 |
| `customer_age_days` | 0.0041 | 0.0012 |
| `avg_order_value` | 0.0035 | 0.0010 |
| `cancellation_ratio` | 0.0020 | 0.0006 |
| `monetary` | 0.0002 | 0.0005 |
| `cancellation_count` | -0.0001 | 0.0001 |

Importance is measured on the test window by shuffling one feature at a time, so it is model-agnostic and not inflated by training-set fit.

## 6. Figures

### ROC curves (all three models)

![ROC curves (all three models)](figures/roc_curves.png)

### Precision-Recall curves

![Precision-Recall curves](figures/pr_curve.png)

### Confusion matrix (LogisticRegression)

![Confusion matrix (LogisticRegression)](figures/confusion_matrix.png)

### Permutation importance

![Permutation importance](figures/feature_importance.png)

### Churn rate by recency bucket

![Churn rate by recency bucket](figures/churn_rate.png)

### RFM distributions by churn status

![RFM distributions by churn status](figures/rfm_by_churn.png)

## 7. Saved artifact

| Item | Value |
|---|---|
| Path | `models/churn_model.joblib` |
| Model | LogisticRegression |
| ROC-AUC | 0.7443 |
| PR-AUC | 0.6342 |
| Trained on | feature window <= 2011-06-12 (2,821 customers) |
| Snapshot date | 2011-06-13 |

The saved artifact is the model trained on the **training** window, i.e. exactly the model whose metrics are reported above, so the numbers can be reproduced by loading it. For a production deployment you would refit on the full feature window (all customers up to the latest cut date) - see the README.

Feature columns expected by the model, in order:

```
recency              # Days between the customer's last purchase and the snapshot date (cut + 1 day).
frequency            # Number of distinct invoices in the feature window.
monetary             # Total revenue (Quantity x UnitPrice) in the feature window.
avg_order_value      # monetary / frequency.
total_items          # Total units purchased in the feature window.
unique_skus          # Number of distinct StockCodes purchased.
customer_age_days    # Days between the customer's first purchase and the cut date.
cancellation_count   # Distinct cancelled invoices (InvoiceNo starting with 'C') before the cut.
cancellation_ratio   # cancellation_count / (cancellation_count + frequency).
```

Score new customers with:

```bash
python predict.py --input data/sample/predict_example.csv
```
