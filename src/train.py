from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import clean_data, load_raw_data

df = clean_data(load_raw_data())

X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Numeric = anything pandas recognises as a number; everything else is text/category.
# Selecting this way avoids depending on how pandas labels string columns.
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# Models only understand numbers. One-hot encoding turns each category
# (e.g. deposit_type = "Non Refund") into its own 0/1 column.
# handle_unknown="ignore" stops the app from crashing later if a user
# enters a category the model never saw in training.
# StandardScaler puts numeric columns on a similar scale. Logistic Regression
# is sensitive to this (lead_time goes up to 700, adults is 1-4).
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

# Bundling preprocessing + model in one Pipeline means the scaler/encoder are
# fit on TRAIN data only (no peeking at test), and the backend can later load
# this single object and feed it raw booking details.
baseline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

baseline.fit(X_train, y_train)

y_pred = baseline.predict(X_test)
y_proba = baseline.predict_proba(X_test)[:, 1]

print("Always-predict-0 accuracy:", round((y_test == 0).mean(), 4))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))