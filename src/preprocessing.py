import pandas as pd


def load_raw_data(path: str = "data/hotel_bookings.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw dataframe and returns a cleaned copy.
    Kept separate from EDA so training and (later) prediction
    can reuse the exact same cleaning steps.
    """
    clean_df = df.copy()

    # Remove clearly invalid/suspicious booking records.
    clean_df = clean_df[clean_df["adults"] <= 10]
    clean_df = clean_df[clean_df["adr"] >= 0]

    # children: only 4 missing, safe to fill with 0.
    clean_df["children"] = clean_df["children"].fillna(0)

    # country: missing means it wasn't recorded — its own category.
    clean_df["country"] = clean_df["country"].fillna("Unknown")

    # agent: missing likely means no agent was involved — 0 as its own category.
    clean_df["agent"] = clean_df["agent"].fillna(0)

    # agent is an ID, not a quantity. Converting to string makes it categorical
    # so the model doesn't think agent 500 > agent 100.
    clean_df["agent"] = clean_df["agent"].astype(int).astype(str)

    # company: ~94% missing, too sparse to use ids directly.
    # Binary flag captures the signal without needing the sparse detail.
    clean_df["has_company"] = clean_df["company"].notna().astype(int)
    clean_df = clean_df.drop(columns=["company"])

        # Drop leakage columns: these describe the final outcome of the booking
    # (e.g. "Canceled" / "Check-Out"), which we would not know when making a
    # real prediction. Keeping them would make the model look perfect but
    # be useless in practice.
    clean_df = clean_df.drop(columns=["reservation_status", "reservation_status_date"])

    # Drop exact duplicate rows BEFORE splitting. If the same row ended up in
    # both train and test, the model would be tested on data it already saw,
    # inflating the score.
    clean_df = clean_df.drop_duplicates()

    # Not known at booking time: the assigned room is decided at check-in and
    # booking_changes counts amendments made after the booking. The app can't
    # ask a user for these, and the leakage check showed dropping them costs
    # almost nothing (ROC-AUC 0.908 -> 0.902).
    clean_df = clean_df.drop(columns=["assigned_room_type", "booking_changes"])

    return clean_df