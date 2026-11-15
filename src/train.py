from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict, train_test_split

from src.preprocessing import build_pipeline, clean_data, load_raw_data

RANDOM_STATE = 42

# ---- Stage A/B/F: load and rule-based cleaning (nothing fitted, so safe before the split) ----
raw = load_raw_data()
df = clean_data(raw)
print("Raw shape:", raw.shape, "-> cleaned shape:", df.shape)

X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

# ---- Stage H: split BEFORE anything is fitted. The pipeline below fits its scaler, encoder and
# feature selectors on X_train only. stratify keeps the cancel ratio equal in both parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape)


def evaluate(name, y_true, pred, proba):
    return {"model": name,
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred),
            "roc_auc": roc_auc_score(y_true, proba)}


# ---- Stage H: compare a simple baseline with a stronger model, same preprocessing for both ----
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                            random_state=RANDOM_STATE),
}
results, pipes = [], {}
for name, model in models.items():
    pipe = build_pipeline(model, RANDOM_STATE)
    pipe.fit(X_train, y_train)
    pipes[name] = pipe
    results.append(evaluate(name, y_test, pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]))

print("\nAlways-predict-0 accuracy:", round((y_test == 0).mean(), 4))
print(pd.DataFrame(results).set_index("model").round(3))

# ---- Stage G: how many features did each step keep? (before / after counts) ----
final_pipe = pipes["Random Forest"]
labels = {"preprocess": "after encoding + binning + scaling", "variance": "after variance threshold",
          "correlation": "after correlation filter", "importance": "after importance selection"}
print("\nFeature counts")
print(f"  raw columns: {raw.shape[1] - 1}")
print(f"  after cleaning: {X_train.shape[1]}")
for i, (step, _) in enumerate(final_pipe.steps[:-1], start=1):
    print(f"  {labels[step]}: {final_pipe[:i].transform(X_train).shape[1]}")

# ---- Threshold tuning on out-of-fold TRAIN predictions (the test set stays untouched) ----
# Every training row gets a probability from a model that never saw it. The whole pipeline,
# including the scaler and selectors, is refitted inside each fold, so there is no leakage.
oof = cross_val_predict(clone(final_pipe), X_train, y_train, cv=3, method="predict_proba")[:, 1]
best_t, best_f1 = 0.5, 0.0
print("\nThreshold tuning (out-of-fold, train only):")
print(f"{'Threshold':<11}{'Precision':>10}{'Recall':>9}{'F1':>7}")
for t in np.arange(0.25, 0.65, 0.05):
    p = (oof >= t).astype(int)
    f1 = f1_score(y_train, p)
    print(f"{t:<11.2f}{precision_score(y_train, p):>10.3f}{recall_score(y_train, p):>9.3f}{f1:>7.3f}")
    if f1 > best_f1:
        best_t, best_f1 = float(t), f1
print(f"\nBest F1 threshold: {best_t:.2f}")

test_proba = final_pipe.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_t).astype(int)
print(f"\nTest set at threshold {best_t:.2f}:")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred))
print("ROC-AUC:", round(roc_auc_score(y_test, test_proba), 4))

# ---- Save: one file with the whole pipeline + decision threshold ----
Path("model").mkdir(exist_ok=True)
joblib.dump({"pipeline": final_pipe, "threshold": best_t, "feature_columns": X.columns.tolist()},
            "model/hotel_cancellation_model.pkl", compress=3)
print("\nModel saved to model/hotel_cancellation_model.pkl")
