from sklearn.model_selection import train_test_split
from src.preprocessing import load_raw_data, clean_data

df = load_raw_data()
clean_df = clean_data(df)

print("Cleaned shape:", clean_df.shape)
print("Missing values remaining:", clean_df.isnull().sum().sum())
print(clean_df["is_canceled"].value_counts(normalize=True) * 100)

# X = inputs the model learns from, y = the answer we want it to predict.
X = clean_df.drop(columns=["is_canceled"])
y = clean_df["is_canceled"]

# Hold out 20% as a test set the model never sees during training.
# stratify=y keeps the 63/37 cancel ratio the same in both halves,
# otherwise a random split could skew the classes.
# random_state makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)