"""Generate a SYNTHETIC monthly stock dataset that mimics a national logistics
system / DHIS2 export. No real facility, patient or stock data is used.

Run:  python src/generate_data.py            (writes to data/)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
DISTRICTS = [
    "Balaka", "Blantyre", "Chikwawa", "Chiradzulu", "Chitipa", "Dedza", "Dowa",
    "Karonga", "Kasungu", "Likoma", "Lilongwe", "Machinga", "Mangochi", "Mchinji",
    "Mulanje", "Mwanza", "Mzimba", "Neno", "Nkhata Bay", "Nkhotakota", "Nsanje",
    "Ntcheu", "Ntchisi", "Phalombe", "Rumphi", "Salima", "Thyolo", "Zomba",
]

# base = typical monthly demand for a SMALL facility; amp = seasonal swing;
# peak = month of the year (Jan = 0) when demand is highest.
MEDICINES = {
    "Artemether-lumefantrine": dict(base=120, amp=0.60, peak=2),   # malaria, rainy season
    "Amoxicillin dispersible": dict(base=80, amp=0.30, peak=6),    # cold season
    "ORS + zinc": dict(base=60, amp=0.45, peak=0),                 # diarrhoea, hot rainy season
    "DMPA (injectable contraceptive)": dict(base=50, amp=0.05, peak=0),  # flat control
}

SIZES = {"small": (1.0, 0.55), "medium": (2.5, 0.33), "large": (6.0, 0.12)}  # (demand x, share)
P_REMOTE_BY_SIZE = {"small": 0.45, "medium": 0.25, "large": 0.05}
LEAD_DAYS = {"urban": (14, 4), "remote": (30, 8)}   # mean, sd of order-to-receipt days
P_LATE = {"urban": 0.10, "remote": 0.30}            # chance a delivery is delayed
P_MISSING_REPORT = {"urban": 0.03, "remote": 0.07}  # chance a monthly report is missing
TARGET_MONTHS = 2.5                                  # order-up-to level, in months of recent use


def make_facilities(rng, n):
    size = rng.choice(list(SIZES), n, p=[v[1] for v in SIZES.values()])
    p_remote = np.array([P_REMOTE_BY_SIZE[s] for s in size])
    remote = rng.random(n) < p_remote
    km = np.where(remote, rng.normal(90, 25, n), rng.normal(20, 10, n)).clip(2, 200)
    return pd.DataFrame({
        "facility_id": [f"FAC-{i:03d}" for i in range(1, n + 1)],
        "district": rng.choice(DISTRICTS, n),
        "facility_size": size,
        "remoteness": np.where(remote, "remote", "urban"),
        "distance_to_depot_km": km.round(0).astype(int),
    })


def make_supply_shocks(rng, n_months):
    """District-level fill-rate shocks: 1.0 normally, 0.3-0.6 during a shortage."""
    shocks = {}
    for d in DISTRICTS:
        for m in MEDICINES:
            fill = np.ones(n_months)
            t = 0
            while t < n_months:
                if rng.random() < 0.04:
                    dur = int(rng.integers(1, 4))
                    fill[t:t + dur] = rng.uniform(0.3, 0.6)
                    t += dur
                t += 1
            shocks[(d, m)] = fill
    return shocks


def simulate(rng, fac, n_months, start):
    shocks = make_supply_shocks(rng, n_months)
    months = pd.period_range(start, periods=n_months, freq="M")
    rows = []
    for f in fac.itertuples():
        mult = SIZES[f.facility_size][0]
        for med, p in MEDICINES.items():
            base = p["base"] * mult
            stock, hist, pipeline, last_lt = 2.0 * base, [base] * 3, {}, np.nan
            for t in range(n_months):
                season = 1 + p["amp"] * np.cos(2 * np.pi * ((t % 12) - p["peak"]) / 12)
                demand = base * season * rng.lognormal(0, 0.2)

                arrivals = pipeline.pop(t, [])
                deliveries = sum(a[0] for a in arrivals)
                if arrivals:
                    last_lt = float(np.mean([a[1] for a in arrivals]))

                available = stock + deliveries
                consumption = min(demand, available)
                unmet = demand - consumption
                days_out = int(round(30 * unmet / demand))
                stock = available - consumption
                hist.append(consumption)

                # place an order at month end (order-up-to ~2.5 months of recent use)
                on_order = sum(a[2] for a in sum(pipeline.values(), []))
                order = max(0.0, TARGET_MONTHS * np.mean(hist[-3:]) - stock - on_order)
                if order > 0:
                    mean, sd = LEAD_DAYS[f.remoteness]
                    lead = max(5.0, rng.normal(mean, sd))
                    if rng.random() < P_LATE[f.remoteness]:
                        lead += rng.uniform(10, 35)
                    arr = t + 1 + int(lead // 30)
                    if arr < n_months:
                        received = order * shocks[(f.district, med)][arr]
                        pipeline.setdefault(arr, []).append((received, lead, order))

                rows.append(dict(
                    facility_id=f.facility_id, district=f.district,
                    facility_size=f.facility_size, remoteness=f.remoteness,
                    distance_to_depot_km=f.distance_to_depot_km, medicine=med,
                    month=str(months[t]), month_index=t,
                    stock_on_hand=round(stock), consumption=round(consumption),
                    deliveries_received=round(deliveries),
                    delivery_lead_time_days=round(last_lt) if not np.isnan(last_lt) else np.nan,
                    stockout_days=days_out, stockout_flag=int(days_out > 0),
                ))
    df = pd.DataFrame(rows)

    # missing monthly reports (all medicines for that facility-month), more often if remote
    keys = df[["facility_id", "month", "remoteness"]].drop_duplicates()
    miss = rng.random(len(keys)) < keys["remoteness"].map(P_MISSING_REPORT).to_numpy()
    keys = keys.assign(reported=(~miss).astype(int))[["facility_id", "month", "reported"]]
    df = df.merge(keys, on=["facility_id", "month"], how="left")
    for col in ["stock_on_hand", "consumption", "deliveries_received",
                "delivery_lead_time_days", "stockout_days", "stockout_flag"]:
        df[col] = df[col].astype("float").where(df["reported"] == 1)
    return df


def to_dhis2_style(df):
    """Long format like a DHIS2 analytics export: one row per data element x period x org unit."""
    value_cols = ["stock_on_hand", "consumption", "deliveries_received", "stockout_days"]
    long = df.melt(id_vars=["facility_id", "medicine", "month"], value_vars=value_cols,
                   var_name="indicator", value_name="value").dropna(subset=["value"])
    long["dataElement"] = long["medicine"] + " | " + long["indicator"]
    long["period"] = long["month"].str.replace("-", "")
    return long.rename(columns={"facility_id": "orgUnit"})[
        ["dataElement", "period", "orgUnit", "value"]].astype({"value": int})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facilities", type=int, default=200)
    ap.add_argument("--months", type=int, default=36)
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    fac = make_facilities(rng, a.facilities)
    df = simulate(rng, fac, a.months, a.start)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fac.to_csv(out / "org_units.csv", index=False)
    df.to_csv(out / "facility_months.csv", index=False)
    to_dhis2_style(df).to_csv(out / "dhis2_style_export.csv", index=False)
    print(f"Wrote {len(df):,} facility-medicine-month rows to {out}/ (SYNTHETIC DATA)")


if __name__ == "__main__":
    main()
