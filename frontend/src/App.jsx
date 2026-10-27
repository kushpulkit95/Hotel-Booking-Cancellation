import { useState } from "react";
import "./App.css";

// Backend address in one constant, so changing it later (deployment)
// is a one-line edit.
const API_URL = "http://127.0.0.1:8000/predict";

// These option lists must match what the model saw in training,
// otherwise the backend rejects the request with a 422.
const DEPOSIT_TYPES = ["No Deposit", "Non Refund", "Refundable"];
const MARKET_SEGMENTS = [
  "Aviation", "Complementary", "Corporate", "Direct",
  "Groups", "Offline TA/TO", "Online TA",
];
const COMMON_COUNTRIES = [
  "PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL",
  "BRA", "NLD", "USA", "CHE", "IND", "CHN", "AUT",
];

// Inputs always hold strings in HTML. We convert these to numbers
// before sending, since the backend expects numbers, not text.
const NUMERIC_FIELDS = [
  "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
  "adults", "previous_cancellations", "total_of_special_requests", "adr",
];

const INITIAL_FORM = {
  hotel: "City Hotel",
  arrival_date: "",
  lead_time: 60,
  stays_in_weekend_nights: 1,
  stays_in_week_nights: 2,
  adults: 2,
  deposit_type: "No Deposit",
  market_segment: "Online TA",
  previous_cancellations: 0,
  total_of_special_requests: 0,
  adr: 100,
  country: "PRT",
};

// Small wrapper so every field has the same label/hint layout.
function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="label">{label}</span>
      {children}
      {hint && <span className="hint">{hint}</span>}
    </label>
  );
}

export default function App() {
  const [form, setForm] = useState(INITIAL_FORM);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // One handler for every input: the input's name matches the form key.
  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault(); // stop the browser from reloading the page
    setLoading(true);
    setError("");
    setResult(null);

    const payload = { ...form, country: form.country.toUpperCase() };
    NUMERIC_FIELDS.forEach((key) => {
      payload[key] = Number(payload[key]);
    });

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok) {
        // FastAPI sends validation problems as a list of {loc, msg}.
        // Show which field is wrong instead of a vague "error".
        const detail = Array.isArray(data.detail)
          ? data.detail.map((d) => `${d.loc.at(-1)}: ${d.msg}`).join("; ")
          : "Request failed";
        throw new Error(detail);
      }
      setResult(data);
    } catch (err) {
      // A TypeError here means the request never reached the API at all.
      setError(
        err instanceof TypeError
          ? "Can't reach the API. Is the backend running on port 8000?"
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const percent = result ? result.cancellation_probability * 100 : 0;

  return (
    <div className="page">
      <h1>Hotel Booking Cancellation Predictor</h1>
      <p className="subtitle">
        Enter the booking details to estimate how likely it is to be cancelled.
      </p>

      <form className="card" onSubmit={handleSubmit}>
        <div className="grid">
          <Field label="Hotel">
            <select name="hotel" value={form.hotel} onChange={handleChange}>
              <option>City Hotel</option>
              <option>Resort Hotel</option>
            </select>
          </Field>

          <Field label="Arrival date">
            <input type="date" name="arrival_date" required
              value={form.arrival_date} onChange={handleChange} />
          </Field>

          <Field label="Lead time (days)" hint="Days between booking and arrival">
            <input type="number" name="lead_time" min="0" max="800" required
              value={form.lead_time} onChange={handleChange} />
          </Field>

          <Field label="Adults">
            <input type="number" name="adults" min="1" max="10" required
              value={form.adults} onChange={handleChange} />
          </Field>

          <Field label="Weekend nights">
            <input type="number" name="stays_in_weekend_nights" min="0" max="20" required
              value={form.stays_in_weekend_nights} onChange={handleChange} />
          </Field>

          <Field label="Week nights">
            <input type="number" name="stays_in_week_nights" min="0" max="50" required
              value={form.stays_in_week_nights} onChange={handleChange} />
          </Field>

          <Field label="Deposit type">
            <select name="deposit_type" value={form.deposit_type} onChange={handleChange}>
              {DEPOSIT_TYPES.map((d) => <option key={d}>{d}</option>)}
            </select>
          </Field>

          <Field label="Market segment">
            <select name="market_segment" value={form.market_segment} onChange={handleChange}>
              {MARKET_SEGMENTS.map((m) => <option key={m}>{m}</option>)}
            </select>
          </Field>

          <Field label="Previous cancellations">
            <input type="number" name="previous_cancellations" min="0" max="30" required
              value={form.previous_cancellations} onChange={handleChange} />
          </Field>

          <Field label="Special requests" hint="0 to 5">
            <input type="number" name="total_of_special_requests" min="0" max="5" required
              value={form.total_of_special_requests} onChange={handleChange} />
          </Field>

          <Field label="Price per night (ADR)">
            <input type="number" name="adr" min="0" max="1000" step="0.01" required
              value={form.adr} onChange={handleChange} />
          </Field>

          <Field label="Country" hint="3-letter code, e.g. PRT, GBR, IND">
            <input name="country" list="countries" maxLength={3} minLength={3} required
              value={form.country} onChange={handleChange} />
            <datalist id="countries">
              {COMMON_COUNTRIES.map((c) => <option key={c} value={c} />)}
            </datalist>
          </Field>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className={`card result ${result.will_cancel ? "bad" : "good"}`}>
          <div className="prob">{percent.toFixed(1)}%</div>
          <div className="prob-label">estimated cancellation probability</div>
          <div className="verdict">{result.prediction}</div>

          {/* The bar shows the probability; the tick shows where the
              model's decision threshold sits, so the verdict is explained. */}
          <div className="bar">
            <div className="bar-fill" style={{ width: `${percent}%` }} />
            <div className="bar-threshold" style={{ left: `${result.threshold * 100}%` }} />
          </div>
          <p className="note">
            Flagged as likely to cancel when the probability is above{" "}
            {(result.threshold * 100).toFixed(0)}%. Fields not shown here use
            typical booking values.
          </p>
        </div>
      )}
    </div>
  );
}