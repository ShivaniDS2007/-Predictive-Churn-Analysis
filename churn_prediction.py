"""
Predictive Churn Analysis
=========================
Builds and evaluates Logistic Regression + Random Forest classifiers
to forecast customer churn probability.

Pipeline:
1. Load data & encode features (One-Hot for categoricals, MinMax for numerics)
2. Train/test split (80/20, stratified)
3. Train Logistic Regression and Random Forest classifiers
4. Evaluate: Precision, Recall, F1-Score, ROC-AUC (+ ROC curve plot)
5. Export churn risk-score predictions to CSV

Run:
    python churn_prediction.py
Outputs land in ./outputs/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)

DATA_PATH = "customer_churn_sample.csv"
OUT_DIR = "outputs"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# Target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop ID column (not a feature)
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

numeric_cols = ["Age", "TenureMonths", "MonthlyCharges", "TotalCharges", "SupportTickets"]
categorical_cols = ["Gender", "SubscriptionType", "ContractType", "PaymentMethod"]

# ---------------------------------------------------------------
# 2. Preprocessing: One-Hot Encoding + MinMax Scaling
# ---------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# ---------------------------------------------------------------
# 3. Train/test split (80/20)
# ---------------------------------------------------------------
# NOTE: This sample dataset has only 15 rows. With stratified split,
# 20% test size on 15 rows is not reliable for statistically robust
# evaluation - metrics below are illustrative of the PIPELINE, not
# a production-grade model. Swap in the full dataset for real results.
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    # Falls back to non-stratified split if a class is too small to stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---------------------------------------------------------------
# 4. Build & train models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
fitted_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # ROC-AUC needs both classes present in y_test
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")

    results[name] = {
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": auc,
    }

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------
# 5. Results summary
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).T
results_df.index.name = "Model"
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv")
print("\n=== Model Comparison ===")
print(results_df)

# Pick best model by F1-Score for the final risk-score export
best_model_name = results_df["F1-Score"].idxmax()
best_pipe = fitted_pipelines[best_model_name]
print(f"\nBest model (by F1-Score): {best_model_name}")

# ---------------------------------------------------------------
# 6. ROC Curve plot (all models)
# ---------------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, pipe in fitted_pipelines.items():
    y_proba = pipe.predict_proba(X_test)[:, 1]
    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})", linewidth=2)
    except ValueError:
        pass
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Churn Prediction Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150)
plt.close()
print(f"Saved ROC curve -> {OUT_DIR}/roc_curve.png")

# ---------------------------------------------------------------
# 7. Export churn risk-score predictions for ALL customers
# ---------------------------------------------------------------
all_proba = best_pipe.predict_proba(X)[:, 1]
risk_df = df[["CustomerID"]].copy()
risk_df["Actual_Churn"] = df["Churn"].map({1: "Yes", 0: "No"})
risk_df["Churn_Risk_Score"] = np.round(all_proba, 4)
risk_df["Risk_Level"] = pd.cut(
    risk_df["Churn_Risk_Score"],
    bins=[-0.01, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"],
)
risk_df = risk_df.sort_values("Churn_Risk_Score", ascending=False)
risk_df.to_csv(f"{OUT_DIR}/churn_risk_predictions.csv", index=False)
print(f"Saved risk-score predictions -> {OUT_DIR}/churn_risk_predictions.csv")

print("\nDone. All outputs are in the 'outputs/' folder.")
