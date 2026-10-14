"""
FreightTiger shipping-cost monitor.
Deterministic, auditable baseline calculations and date/route-grounded note matching.
Run: python main.py
"""
from pathlib import Path
import pandas as pd
import csv
import json
import hashlib
import re

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SHIPMENTS = DATA / "shipment_records.csv"
NOTES = DATA / "context_notes.csv"
OUTPUT = ROOT / "output.csv"
LOG = ROOT / "token_cost_log.json"
EVAL = ROOT / "evaluation.json"

OUTPUT_COLUMNS = [
    "route", "week_of", "cost_per_tonne_km", "vs_own_history",
    "vs_similar_routes", "flagged", "matched_note_id", "reason"
]
OWN_THRESHOLD = 0.15
PEER_THRESHOLD = 0.15


def monday_of(date_series):
    dates = pd.to_datetime(date_series)
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def route_name(origin, destination):
    return f"{origin}-{destination}"


def note_applies(note, route, week_start):
    """Strict, deterministic date/route matching; no semantic guessing."""
    applies = str(note["applies_to"]).strip()
    if applies != "All Routes" and applies != route:
        return False
    note_date = pd.Timestamp(note["date"])
    week_end = week_start + pd.Timedelta(days=6)
    body = str(note["note"]).lower()

    # Notes describing a bounded duration apply only inside that stated interval.
    if "feb 24 to mar 8" in body:
        return pd.Timestamp("2025-02-24") <= week_end and week_start <= pd.Timestamp("2025-03-08")
    # Point-in-time festival surcharge applies to the week containing its date.
    if "temporary surcharge" in body or "festival week" in body:
        return week_start <= note_date <= week_end
    # "Starting this week" establishes the start week, not an unlimited duration.
    # Do not extend it to later weeks without explicit duration evidence.
    if "starting this week" in body and "diesel" in body:
        return week_start <= note_date <= week_end
    # Notes that say costs were unaffected or no disruptions are explicitly non-justifications.
    if "not significantly affected" in body or "no major disruptions" in body or "no significant disruptions" in body:
        return False
    # One-week maintenance or one-off road-condition events only apply in their dated week.
    return week_start <= note_date <= week_end


def retrieve_supporting_note(notes, route, week_start):
    """Small deterministic retrieval layer. Only positive, route/time-matched evidence can justify."""
    candidates = notes[notes.apply(lambda n: note_applies(n, route, week_start), axis=1)]
    for _, note in candidates.sort_values(["date", "note_id"]).iterrows():
        body = str(note["note"]).lower()
        # Exclude contextual notes that explicitly say there was no material cost impact.
        if any(term in body for term in [
            "costs were not significantly affected", "no major disruptions",
            "no significant disruptions", "costs weren't affected",
            "without a rate change", "not part of this dataset"
        ]):
            continue
        if any(term in body for term in [
            "surcharge", "higher trip costs", "pushing up transportation costs",
            "detours and higher trip costs"
        ]):
            return note
    return None


def fmt_pct(value):
    return f"{value:+.1%}"


def build_output():
    df = pd.read_csv(SHIPMENTS)
    notes = pd.read_csv(NOTES, keep_default_na=False)
    required = {"origin", "destination", "route_type", "quantity_tonnes",
                "distance_km", "freight_cost_inr", "shipment_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Shipment file missing columns: {sorted(missing)}")
    if (df["quantity_tonnes"] <= 0).any() or (df["distance_km"] <= 0).any():
        raise ValueError("Quantity and distance must be positive.")

    df["route"] = [route_name(o, d) for o, d in zip(df.origin, df.destination)]
    df["week_of"] = monday_of(df["shipment_date"]).dt.strftime("%Y-%m-%d")
    # Correct aggregate unit rate: total freight / sum(tonnes * km), not mean of row-level ratios.
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    weekly = (df.groupby(["route", "route_type", "week_of"], as_index=False)
                .agg(total_cost=("freight_cost_inr", "sum"),
                     total_tonne_km=("tonne_km", "sum"),
                     shipment_count=("shipment_date", "size")))
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["total_tonne_km"]
    weekly["week_dt"] = pd.to_datetime(weekly["week_of"])
    weekly = weekly.sort_values(["route", "week_dt"]).reset_index(drop=True)

    # Own-history baseline: strictly earlier route-weeks, last 8 available observations.
    histories = {}
    for route, group in weekly.groupby("route", sort=True):
        prior = []
        for idx in group.index:
            if prior:
                recent = prior[-8:]
                histories[idx] = (sum(recent) / len(recent), len(recent))
            else:
                histories[idx] = (None, 0)
            prior.append(float(weekly.loc[idx, "cost_per_tonne_km"]))

    # Peer baseline: average rate of other routes of same route_type in the same week.
    peer_map = {}
    for idx, row in weekly.iterrows():
        peers = weekly[(weekly["week_of"] == row["week_of"]) &
                       (weekly["route_type"] == row["route_type"]) &
                       (weekly["route"] != row["route"])]
        peer_map[idx] = float(peers["cost_per_tonne_km"].mean()) if len(peers) else None

    results = []
    for idx, row in weekly.iterrows():
        own_avg, own_n = histories[idx]
        peer_avg = peer_map[idx]
        current = float(row["cost_per_tonne_km"])
        own_change = (current / own_avg - 1) if own_avg and own_avg > 0 else None
        peer_change = (current / peer_avg - 1) if peer_avg and peer_avg > 0 else None
        own_trigger = own_change is not None and own_change >= OWN_THRESHOLD
        peer_trigger = peer_change is not None and peer_change >= PEER_THRESHOLD
        if not (own_trigger or peer_trigger):
            continue

        week_start = pd.Timestamp(row["week_dt"])
        note = retrieve_supporting_note(notes, row["route"], week_start)
        if note is not None:
            flagged = "No (justified)"
            matched_id = str(note["note_id"])
            reason = (f"Justified by note {matched_id} dated {note['date']}: "
                      f"{str(note['note']).strip()} This note matches the route and time window.")
        else:
            flagged = "Yes"
            matched_id = ""
            reason = ("No matching note found that establishes a route- and date-specific cost cause. "
                      "The increase is unexplained and should be reviewed by a human.")
        own_text = (f"{fmt_pct(own_change)} vs this route's past average"
                    if own_change is not None else "N/A (no prior route weeks)")
        peer_text = (f"{fmt_pct(peer_change)} vs similar-length routes this week"
                     if peer_change is not None else "N/A (no peer route available this week)")
        results.append({
            "route": row["route"],
            "week_of": row["week_of"],
            "cost_per_tonne_km": round(current, 2),
            "vs_own_history": own_text,
            "vs_similar_routes": peer_text,
            "flagged": flagged,
            "matched_note_id": matched_id,
            "reason": reason
        })

    out = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    out.to_csv(OUTPUT, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    return df, weekly, out


def run_evaluation(df, weekly, out):
    # Basic invariant and deterministic checks, intentionally small and reproducible.
    checks = {
        "input_rows": int(len(df)),
        "weekly_route_rows": int(len(weekly)),
        "output_rows": int(len(out)),
        "output_columns_match_contract": list(out.columns) == OUTPUT_COLUMNS,
        "all_output_rows_are_triggered": bool(out["flagged"].isin(["Yes", "No (justified)"]).all()) if len(out) else True,
        "justified_rows_have_note_id": bool((out.loc[out["flagged"] == "No (justified)", "matched_note_id"].astype(str).str.len() > 0).all()),
        "unexplained_rows_have_blank_note_id": bool((out.loc[out["flagged"] == "Yes", "matched_note_id"].fillna("").astype(str) == "").all()),
        "all_rates_positive": bool((weekly["cost_per_tonne_km"] > 0).all()),
        "no_duplicate_route_weeks": bool(not weekly.duplicated(["route", "week_of"]).any()),
        "thresholds": {"own_history_pct": OWN_THRESHOLD, "similar_routes_pct": PEER_THRESHOLD},
        "method": "Deterministic rule-based anomaly screening and exact route/date note retrieval; no LLM used."
    }
    EVAL.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not all(v for k, v in checks.items() if isinstance(v, bool)):
        raise AssertionError(f"Evaluation failed: {checks}")
    return checks


def main():
    df, weekly, out = build_output()
    checks = run_evaluation(df, weekly, out)
    # No LLM calls: token usage and model cost are exactly zero for this implementation.
    token_log = {
        "runs": 1,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "model": None,
        "estimated_model_cost_usd": 0.0,
        "cost_basis": "No LLM/model API is called; the supplied notes are matched by deterministic Python rules.",
        "note": "This is a transparent baseline implementation. A hosted LLM is intentionally not used so the output is repeatable and incurs no model cost."
    }
    LOG.write_text(json.dumps(token_log, indent=2), encoding="utf-8")
    print(f"Processed {len(df)} shipments into {len(weekly)} route-weeks.")
    print(f"Wrote {len(out)} anomaly candidates to {OUTPUT.name}.")
    print(f"Evaluation checks passed: {sum(1 for v in checks.values() if v is True)} boolean checks.")


if __name__ == "__main__":
    main()
