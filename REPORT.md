# Predictive Churn Analysis — Report

## 1. Objective
Build a classification pipeline that forecasts customer churn probability using
Logistic Regression and Random Forest, so at-risk customers can be flagged for
retention action.

## 2. Dataset
- Source file: `customer_churn_sample.csv`
- 15 customers, 11 columns
- Target: `Churn` (Yes/No)
- Features used: `Age`, `TenureMonths`, `SubscriptionType`, `MonthlyCharges`,
  `TotalCharges`, `ContractType`, `SupportTickets`, `PaymentMethod`, `Gender`
- `CustomerID` dropped (identifier, not predictive)

> **Note on dataset size:** This sample has only 15 rows. That's enough to prove
> the pipeline works end-to-end, but not enough for statistically meaningful
> evaluation (test set = 3 rows). For the actual submission, swap in the full
> churn dataset (e.g. the ~7,000-row Telco Customer Churn dataset from Kaggle)
> and re-run the exact same script — no code changes needed, just replace
> `customer_churn_sample.csv`.

## 3. Preprocessing
- **Numeric features** (`Age`, `TenureMonths`, `MonthlyCharges`, `TotalCharges`,
  `SupportTickets`) → scaled with `MinMaxScaler` (0–1 range).
- **Categorical features** (`Gender`, `SubscriptionType`, `ContractType`,
  `PaymentMethod`) → converted with `OneHotEncoder`.
- Combined via `ColumnTransformer` inside an sklearn `Pipeline`, so
  preprocessing + model are fit together and reusable on new data without
  leakage.

## 4. Train/Test Split
- 80/20 split, stratified on `Churn` (falls back to a plain split if a class
  is too small to stratify).
- `random_state=42` for reproducibility.

## 5. Models Trained
1. **Logistic Regression** (`max_iter=1000`)
2. **Random Forest** (`n_estimators=200`)

Both trained inside the same preprocessing pipeline for a fair comparison.

## 6. Evaluation Metrics
| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 |

(See `outputs/model_comparison.csv` for the machine-readable version.)

Both models score perfectly here because the sample data is small and the
churn pattern is very clean (short-tenure, month-to-month customers churn;
long-tenure, annual/two-year contract customers don't). On the full dataset,
expect meaningfully lower — and more realistic — scores.

ROC curve for both models: `outputs/roc_curve.png`

## 7. Churn Risk-Score Export
`outputs/churn_risk_predictions.csv` contains, for every customer:
- `Churn_Risk_Score` — predicted probability of churn (0–1), from the
  best-performing model (selected by F1-Score)
- `Risk_Level` — bucketed as Low / Medium / High
- `Actual_Churn` — ground truth, for validation

This is the file a retention team would use to prioritize outreach.

## 8. How to Reproduce
```bash
pip install pandas scikit-learn matplotlib
python churn_prediction.py
```
All outputs are written to `./outputs/`.

## 9. Files in this submission
```
churn_prediction.py          # full pipeline script
customer_churn_sample.csv    # input data
REPORT.md                    # this report
outputs/
  model_comparison.csv       # precision/recall/f1/roc-auc per model
  roc_curve.png              # ROC curve plot
  churn_risk_predictions.csv # per-customer churn risk scores
```
