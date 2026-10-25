import calendar
import json
from datetime import date
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent

# Load once at startup, not per request. Loading an 80 MB model on every
# prediction would make the API painfully slow.
bundle = joblib.load(ROOT / "model" / "hotel_cancellation_model.pkl")
pipeline = bundle["pipeline"]
threshold = bundle["threshold"]
feature_columns = bundle["feature_columns"]
defaults = json.loads((ROOT / "model" / "defaults.json").read_text())

app = FastAPI(title="Hotel Booking Cancellation Predictor")

# The React dev server runs on a different port than this API. Browsers
# block cross-origin requests unless the API explicitly allows them.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class BookingInput(BaseModel):
    # Literal restricts values to exactly what the model saw in training.
    # Field(ge=, le=) rejects impossible numbers before they reach the model.
    hotel: Literal["City Hotel", "Resort Hotel"]
    arrival_date: date
    lead_time: int = Field(ge=0, le=800)
    stays_in_weekend_nights: int = Field(ge=0, le=20)
    stays_in_week_nights: int = Field(ge=0, le=50)
    adults: int = Field(ge=1, le=10)
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    market_segment: Literal[
        "Aviation", "Complementary", "Corporate", "Direct",
        "Groups", "Offline TA/TO", "Online TA",
    ]
    previous_cancellations: int = Field(ge=0, le=30)
    total_of_special_requests: int = Field(ge=0, le=5)
    adr: float = Field(ge=0, le=1000)
    country: str = Field(min_length=3, max_length=3)  # e.g. PRT, GBR, IND


@app.get("/")
def root():
    # Friendly pointer so opening the base URL doesn't look broken.
    return {"message": "Hotel Cancellation API is running. Go to /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(booking: BookingInput):
    # Start from the defaults, then overwrite with what the user provided.
    row = dict(defaults)
    d = booking.arrival_date
    row.update({
        "hotel": booking.hotel,
        "lead_time": booking.lead_time,
        # Derived from the single date the user picks. The training data
        # stores the month as a name ("July"), so we match that format.
        "arrival_date_year": d.year,
        "arrival_date_month": calendar.month_name[d.month],
        "arrival_date_week_number": d.isocalendar().week,
        "arrival_date_day_of_month": d.day,
        "stays_in_weekend_nights": booking.stays_in_weekend_nights,
        "stays_in_week_nights": booking.stays_in_week_nights,
        "adults": booking.adults,
        "deposit_type": booking.deposit_type,
        "market_segment": booking.market_segment,
        "previous_cancellations": booking.previous_cancellations,
        "total_of_special_requests": booking.total_of_special_requests,
        "adr": booking.adr,
        "country": booking.country.upper(),
    })

    # One-row DataFrame in the same column order as training.
    X = pd.DataFrame([row])[feature_columns]

    probability = float(pipeline.predict_proba(X)[0, 1])
    will_cancel = probability >= threshold

    return {
        "cancellation_probability": round(probability, 4),
        "will_cancel": will_cancel,
        "prediction": "LIKELY TO BE CANCELLED" if will_cancel else "LIKELY TO BE KEPT",
        "threshold": threshold,
    }