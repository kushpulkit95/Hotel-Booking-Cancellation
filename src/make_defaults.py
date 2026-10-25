import json
from pathlib import Path

from src.preprocessing import clean_data, load_raw_data

# Fields the user types in, plus the ones we derive from the arrival date.
# Everything else needs a default value.
USER_OR_DERIVED = {
    "hotel", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "deposit_type", "market_segment", "previous_cancellations",
    "total_of_special_requests", "adr", "country",
    "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month",
}

X = clean_data(load_raw_data()).drop(columns=["is_canceled"])

defaults = {}
for col in X.columns:
    if col in USER_OR_DERIVED:
        continue
    if X[col].dtype == "object" or str(X[col].dtype) == "str":
        # Categorical: most common value is the safest "typical" guess.
        defaults[col] = str(X[col].mode()[0])
    else:
        # Numeric: median is robust to outliers (unlike the mean).
        defaults[col] = float(X[col].median())

Path("model").mkdir(exist_ok=True)
Path("model/defaults.json").write_text(json.dumps(defaults, indent=2))
print(json.dumps(defaults, indent=2))