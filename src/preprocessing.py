import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column groups. Keeping them in one place means train.py, the notebook and the
# backend all agree on which column gets which treatment.
# ---------------------------------------------------------------------------

# Stage C: ordinal encodings with the order stated explicitly.
MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
MONTH_MAP = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
# Ordered by how many meals are included: none < breakfast < half board < full board.
MEAL_ORDER = {"Undefined": 0, "SC": 0, "BB": 1, "HB": 2, "FB": 3}

# Stage C: nominal columns -> one-hot (no natural order).
NOMINAL_COLS = ["deposit_type", "market_segment", "distribution_channel", "customer_type",
                "reserved_room_type", "country", "agent"]

# Stage D: domain-defined lead-time bins (last-minute, month, quarter, half-year, year, longer).
LEAD_EDGES = [-1, 7, 30, 90, 180, 365, np.inf]

# Stage E: columns to standardise, and the skewed ones (train skewness > 1, checked in the
# notebook) that get a log1p transform first.
CONTINUOUS_COLS = ["lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
                   "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
                   "previous_cancellations", "previous_bookings_not_canceled", "days_in_waiting_list",
                   "adr", "required_car_parking_spaces", "total_of_special_requests"]
ORDINAL_COLS = ["arrival_date_month", "meal", "lead_time_bin", "prev_cancel_bin"]
LOG_COLS = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "children", "babies",
            "previous_cancellations", "previous_bookings_not_canceled", "days_in_waiting_list",
            "required_car_parking_spaces", "total_of_special_requests"]


def load_raw_data(path: str = "data/hotel_bookings.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Rule-based cleaning. Nothing here learns a statistic from the data (only fixed rules and
    # constants), so it is safe to run BEFORE the train/test split.
    d = df.copy()

    # Stage F: remove values that are errors, not genuine extremes.
    # 20-55 adults in one booking, a negative price, and a 5,400 price (next highest is 510).
    d = d[(d["adults"] <= 10) & (d["adr"] >= 0) & (d["adr"] <= 1000)].copy()

    # Leakage: these record how the booking ended, which we don't know at prediction time.
    d = d.drop(columns=["reservation_status", "reservation_status_date"])

    # Exact duplicates could land in both train and test and inflate the score.
    d = d.drop_duplicates()

    # Stage B: constant fills (a constant is not "fitted", so no leakage).
    d["children"] = d["children"].fillna(0)
    d["country"] = d["country"].fillna("Unknown")
    # agent is an ID, not a quantity -> keep it as text so it is treated as a category.
    d["agent"] = d["agent"].fillna(0).astype(int).astype(str)
    # company is ~94% missing: keep only the useful signal "was a company involved?".
    d["has_company"] = d["company"].notna().astype(int)
    d = d.drop(columns=["company"])

    # Not known at booking time (assigned_room_type, booking_changes) or cannot generalise to
    # future dates (the data only covers 2015-2017).
    d = d.drop(columns=["assigned_room_type", "booking_changes", "arrival_date_year"])
    return d.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline steps. Everything below that LEARNS something (scaler, encoder, selectors) is fitted
# only on the training data when you call pipeline.fit(X_train, y_train).
# ---------------------------------------------------------------------------

def add_bins(d: pd.DataFrame) -> pd.DataFrame:
    # Stage D: domain-defined bins. Fixed edges, so nothing is fitted.
    d = d.copy()
    d["lead_time_bin"] = pd.cut(d["lead_time"], bins=LEAD_EDGES, labels=False).astype(int)
    d["prev_cancel_bin"] = pd.cut(d["previous_cancellations"], bins=[-1, 0, 1, np.inf],
                                  labels=False).astype(int)
    d["waiting_list_flag"] = (d["days_in_waiting_list"] > 0).astype(int)
    return d


def encode_binary_ordinal(d: pd.DataFrame) -> pd.DataFrame:
    # Stage C: binary (0/1) and ordinal (explicit order) columns.
    d = d.copy()
    d["hotel"] = d["hotel"].map({"City Hotel": 0, "Resort Hotel": 1})
    d["arrival_date_month"] = d["arrival_date_month"].map(MONTH_MAP)
    d["meal"] = d["meal"].map(MEAL_ORDER)
    return d


def log_transform(d: pd.DataFrame) -> pd.DataFrame:
    # Stage E: log1p pulls in the long right tail of skewed columns (and handles zeros).
    d = d.copy()
    d[LOG_COLS] = np.log1p(d[LOG_COLS])
    return d


class CorrelationFilter(BaseEstimator, TransformerMixin):
    # Stage G: for every pair of columns with |correlation| above the threshold, drop the later
    # one. Near-duplicate columns add no information. Fitted on training data only.
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = pd.DataFrame(X).corr().abs()
        self.columns_in_ = list(corr.columns)   # remembered so output column names can be rebuilt
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.to_drop_ = [c for c in upper.columns if (upper[c] > self.threshold).any()]
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_)

    def get_feature_names_out(self, input_features=None):
        names = self.columns_in_ if input_features is None else input_features
        return np.array([c for c in names if c not in self.to_drop_])


def build_pipeline(model, random_state: int = 42) -> Pipeline:
    # One object: raw cleaned booking in -> prediction out. The backend loads this single pipeline,
    # so training and prediction can never use different preprocessing.
    scale_cols = CONTINUOUS_COLS + ORDINAL_COLS
    prep = Pipeline([
        ("bins", FunctionTransformer(add_bins)),
        ("encode", FunctionTransformer(encode_binary_ordinal)),
        ("log", FunctionTransformer(log_transform)),
        ("columns", ColumnTransformer(
            [("scale", StandardScaler(), scale_cols),
             # Categories under 1% of training rows are grouped into one "infrequent" column,
             # so country/agent don't create hundreds of mostly-empty columns.
             ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                      sparse_output=False), NOMINAL_COLS)],
            remainder="passthrough",              # 0/1 flags stay as they are
            verbose_feature_names_out=False)),
    ])
    pipe = Pipeline([
        ("preprocess", prep),
        ("variance", VarianceThreshold(threshold=0.01)),
        ("correlation", CorrelationFilter(threshold=0.9)),
        ("importance", SelectFromModel(
            RandomForestClassifier(n_estimators=100, min_samples_leaf=3, n_jobs=-1,
                                   random_state=random_state),
            threshold="median")),
        ("model", model),
    ])
    pipe.set_output(transform="pandas")
    return pipe
