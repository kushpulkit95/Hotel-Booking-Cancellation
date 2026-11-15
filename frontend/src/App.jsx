import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/predict";

// [value sent to the API, label shown to the user]
const COUNTRIES = [
  ["PRT", "Portugal"], ["GBR", "United Kingdom"], ["FRA", "France"], ["ESP", "Spain"],
  ["DEU", "Germany"], ["ITA", "Italy"], ["IRL", "Ireland"], ["NLD", "Netherlands"],
  ["BEL", "Belgium"], ["CHE", "Switzerland"], ["USA", "United States"], ["BRA", "Brazil"],
  ["IND", "India"], ["CHN", "China"],
];
const DEPOSITS = [
  ["No Deposit", "No deposit paid"],
  ["Refundable", "Refundable deposit"],
  ["Non Refund", "Non-refundable deposit"],
];
const SEGMENTS = [
  ["Online TA", "Online travel agency (booking websites)"],
  ["Offline TA/TO", "Offline travel agent or tour operator"],
  ["Direct", "Direct with the hotel"],
  ["Corporate", "Company / corporate booking"],
  ["Groups", "Group booking"],
  ["Aviation", "Airline crew"],
  ["Complementary", "Complimentary stay"],
];

const toISO = (d) => d.toISOString().slice(0, 10);
const addDays = (iso, n) => {
  const d = new Date(iso);
  d.setUTCDate(d.getUTCDate() + n);
  return toISO(d);
};
const daysBetween = (a, b) => Math.round((new Date(b) - new Date(a)) / 86400000);
const TODAY = toISO(new Date());

const BLANK = {
  hotel: "City Hotel", bookingDate: TODAY, arrivalDate: addDays(TODAY, 30),
  weekendNights: 1, weekNights: 2, adults: 2, country: "PRT", otherCountry: "",
  previousCancellations: 0, deposit: "No Deposit", segment: "Online TA",
  requests: 0, price: 100,
};

// One-click examples so a first-time user can see a result immediately.
const EXAMPLES = [
  { label: "Higher-risk booking", values: {
    hotel: "City Hotel", arrivalDate: addDays(TODAY, 200), deposit: "Non Refund",
    segment: "Online TA", requests: 0, country: "PRT" } },
  { label: "Safer booking", values: {
    hotel: "Resort Hotel", arrivalDate: addDays(TODAY, 5), deposit: "No Deposit",
    segment: "Direct", requests: 2, country: "GBR" } },
];

const ADVICE = {
  high: ["Ask for a deposit or a card guarantee.", "Send a confirmation message before arrival.",
         "Consider a small overbooking buffer for these dates."],
  medium: ["Send a friendly reminder closer to arrival.", "No other action needed yet."],
  low: ["This booking looks stable. No extra action needed."],
};

function riskLevel(p, threshold) {
  if (p >= threshold) return { key: "high", label: "High risk" };
  if (p >= 0.2) return { key: "medium", label: "Medium risk" };
  return { key: "low", label: "Low risk" };
}

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
  const [form, setForm] = useState(BLANK);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const resultRef = useRef(null);

  const set = (e) => setForm({ ...form, [e.target.name]: e.target.value });
  // Lead time = days between booking and arrival. Computed so the user never has to work it out.
  const leadTime = daysBetween(form.bookingDate, form.arrivalDate);
  const countryCode = form.country === "OTHER" ? form.otherCountry.toUpperCase() : form.country;

  useEffect(() => {
    if (result) resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [result]);

  const applyExample = (values) => {
    setForm({ ...BLANK, ...values });
    setResult(null);
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);

    if (leadTime < 0) return setError("The arrival date must be on or after the booking date.");
    if (leadTime > 800) return setError("Booking and arrival dates more than 800 days apart are not supported.");
    if (Number(form.weekendNights) + Number(form.weekNights) < 1) return setError("Please enter at least one night.");
    if (countryCode.length !== 3) return setError("Please enter a 3-letter country code, for example FRA.");

    setError("");
    setLoading(true);
    const payload = {
      hotel: form.hotel, arrival_date: form.arrivalDate, lead_time: leadTime,
      stays_in_weekend_nights: Number(form.weekendNights), stays_in_week_nights: Number(form.weekNights),
      adults: Number(form.adults), deposit_type: form.deposit, market_segment: form.segment,
      previous_cancellations: Number(form.previousCancellations),
      total_of_special_requests: Number(form.requests), adr: Number(form.price), country: countryCode,
    };

    try {
      const res = await fetch(API_URL, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((d) => `${d.loc.at(-1)}: ${d.msg}`).join("; ") : "Request failed";
        throw new Error(detail);
      }
      setResult(data);
    } catch (err) {
      setError(err instanceof TypeError
        ? "Cannot reach the prediction service. Start the backend (uvicorn) and try again." : err.message);
    } finally {
      setLoading(false);
    }
  };

  const percent = result ? result.cancellation_probability * 100 : 0;
  const level = result ? riskLevel(result.cancellation_probability, result.threshold) : null;

  return (
    <div className="app">
      <header className="hero">
        <h1>Will this booking be cancelled?</h1>
        <p>Enter a few details about a hotel booking and get an instant estimate of how likely
          it is to be cancelled, so you can plan ahead.</p>
      </header>

      <main className="wrap">
        <ol className="steps">
          <li><b>1</b> Fill in the booking details</li>
          <li><b>2</b> Click the estimate button</li>
          <li><b>3</b> Read the risk level and tips</li>
        </ol>

        <div className="examples">
          <span>New here? Try an example:</span>
          {EXAMPLES.map((ex) => (
            <button type="button" key={ex.label} className="chip" onClick={() => applyExample(ex.values)}>
              {ex.label}
            </button>
          ))}
          <button type="button" className="chip ghost" onClick={() => applyExample({})}>Reset</button>
        </div>

        <form onSubmit={handleSubmit}>
          <section className="card">
            <h2><span className="num">1</span> The Stay</h2>
            <div className="grid">
              <Field label="Hotel type" hint="City hotels cancel more often than resorts in this data">
                <select name="hotel" value={form.hotel} onChange={set}>
                  <option>City Hotel</option><option>Resort Hotel</option>
                </select>
              </Field>
              <Field label="Price per night" hint="Average room price per night (typical: 50 to 200)">
                <input type="number" name="price" min="0" max="1000" step="0.01" required
                  value={form.price} onChange={set} />
              </Field>
              <Field label="Booking was made on">
                <input type="date" name="bookingDate" required value={form.bookingDate} onChange={set} />
              </Field>
              <Field label="Guest arrives on"
                hint={leadTime >= 0 ? `Booked ${leadTime} day${leadTime === 1 ? "" : "s"} before arrival` : "Must be after the booking date"}>
                <input type="date" name="arrivalDate" required value={form.arrivalDate} onChange={set} />
              </Field>
              <Field label="Weekend nights" hint="Saturday and Sunday nights">
                <input type="number" name="weekendNights" min="0" max="20" required
                  value={form.weekendNights} onChange={set} />
              </Field>
              <Field label="Weekday nights" hint="Monday to Friday nights">
                <input type="number" name="weekNights" min="0" max="50" required
                  value={form.weekNights} onChange={set} />
              </Field>
            </div>
          </section>

          <section className="card">
            <h2><span className="num">2</span> The Guest</h2>
            <div className="grid">
              <Field label="Number of adults">
                <input type="number" name="adults" min="1" max="10" required value={form.adults} onChange={set} />
              </Field>
              <Field label="Guest's country">
                <select name="country" value={form.country} onChange={set}>
                  {COUNTRIES.map(([code, name]) => <option key={code} value={code}>{name}</option>)}
                  <option value="OTHER">Other (enter code)</option>
                </select>
              </Field>
              {form.country === "OTHER" && (
                <Field label="3-letter country code" hint="For example FRA, JPN, AUS">
                  <input name="otherCountry" maxLength={3} required value={form.otherCountry} onChange={set} />
                </Field>
              )}
              <Field label="Earlier cancellations by this guest" hint="How many past bookings they cancelled">
                <input type="number" name="previousCancellations" min="0" max="30" required
                  value={form.previousCancellations} onChange={set} />
              </Field>
            </div>
          </section>

          <section className="card">
            <h2><span className="num">3</span> The Booking</h2>
            <div className="grid">
              <Field label="Deposit" hint="Non-refundable bookings behave very differently in this data">
                <select name="deposit" value={form.deposit} onChange={set}>
                  {DEPOSITS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
              </Field>
              <Field label="How was it booked?">
                <select name="segment" value={form.segment} onChange={set}>
                  {SEGMENTS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
              </Field>
              <Field label="Special requests" hint="Late check-in, extra bed, quiet room... (0 to 5)">
                <input type="number" name="requests" min="0" max="5" required value={form.requests} onChange={set} />
              </Field>
            </div>
          </section>

          <button type="submit" className="primary" disabled={loading}>
            {loading ? "Calculating..." : "Estimate Cancellation Risk"}
          </button>
        </form>

        {error && <div className="error" role="alert">{error}</div>}

        {result && (
          <section className={`card result ${level.key}`} ref={resultRef}>
            <div className="badge">{level.label}</div>
            <div className="big">{percent.toFixed(0)}%</div>
            <p className="meaning">
              Roughly {Math.round(percent)} out of 100 bookings with these details would be cancelled,
              according to the model.
            </p>

            <div className="meter" aria-hidden="true">
              <div className="zone low" /><div className="zone medium" /><div className="zone high" />
              <div className="marker" style={{ left: `${Math.min(percent, 100)}%` }} />
            </div>
            <div className="meter-labels"><span>Low</span><span>Medium</span><span>High</span></div>

            <h3>What you can do</h3>
            <ul>{ADVICE[level.key].map((t) => <li key={t}>{t}</li>)}</ul>
          </section>
        )}

        <details className="about">
          <summary>About this estimate</summary>
          <ul>
            <li>It is a statistical estimate based on past bookings, not a certainty.</li>
            <li>The model learned from 2015 to 2017 bookings at two hotels in Portugal, so treat it as a guide for similar hotels.</li>
            <li>Details not asked here (such as meal plan or booking agent) are filled in with typical values.</li>
            <li>A booking is flagged as high risk when its estimate is above {result ? Math.round(result.threshold * 100) : 35}%.</li>
          </ul>
        </details>
      </main>
    </div>
  );
}
