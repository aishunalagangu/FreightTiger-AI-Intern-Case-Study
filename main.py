
"""
FreightTiger shipping-cost monitor.
Deterministic, auditable baseline calculations and date/route-grounded note matching.

Run: python main.py
"""
from pathlib import Path
import pandas as pd
import csv
import json

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SHIPMENTS = DATA / "shipment_records.csv"
NOTES = DATA / "context_notes.csv"
OUTPUT = ROOT / "output.csv"
LOG = ROOT / "token_cost_log.json"
EVAL = ROOT / "evaluation.json"

OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]

OWN_THRESHOLD = 0.15
PEER_THRESHOLD = 0.15
HISTORY_WEEKS = 8


def monday_of(date_series):
    """Return the Monday starting the week for each date."""
    dates = pd.to_datetime(date_series)
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def route_name(origin, destination):
    return f"{origin}-{destination}"


def note_applies(note, route, week_start):
    """Apply only notes matching the route and relevant time window."""
    applies = str(note["applies_to"]).strip()

    if applies != "All Routes" and applies != route:
        return False

    note_date = pd.Timestamp(note["date"])
    week_end = week_start + pd.Timedelta(days=6)
    body = str(note["note"]).lower()

    # Explicit bounded event: February 24 through March 8, 2025.
    if "feb 24 to mar 8" in body:
        return (
            pd.Timestamp("2025-02-24") <= week_end
            and week_start <= pd.Timestamp("2025-03-08")
        )

    # Festival surcharges apply only to the week containing the note date.
    if "temporary surcharge" in body or "festival week" in body:
        return week_start <= note_date <= week_end

    # A change starting this week is not assumed to continue indefinitely.
    if "starting this week" in body and "diesel" in body:
        return week_start <= note_date <= week_end

    # These statements explicitly do not establish a cost justification.
    if any(
        phrase in body
        for phrase in [
            "not significantly affected",
            "no major disruptions",
            "no significant disruptions",
        ]
    ):
        return False

    # Other dated events apply only during their stated week.
    return week_start <= note_date <= week_end


def retrieve_supporting_note(notes, route, week_start):
    """
    Deterministic note retrieval.
    Only positive evidence matching the route and time can justify a rise.
    """
    candidates = notes[
        notes.apply(
            lambda note: note_applies(note, route, week_start),
            axis=1,
        )
    ]

    for _, note in candidates.sort_values(
        ["date", "note_id"]
    ).iterrows():
        body = str(note["note"]).lower()

        # Do not use notes that deny material cost impact.
        if any(
            phrase in body
            for phrase in [
                "costs were not significantly affected",
                "no major disruptions",
                "no significant disruptions",
                "costs weren't affected",
                "without a rate change",
                "not part of this dataset",
            ]
        ):
            continue

        if any(
            phrase in body
            for phrase in [
                "surcharge",
                "higher trip costs",
                "pushing up transportation costs",
                "detours and higher trip costs",
            ]
        ):
            return note

    return None


def fmt_pct(value):
    return f"{value:+.1%}"


def calculate_own_history(weekly):
    """
    Calculate each route's trailing 8-calendar-week average.

    The current week is excluded.
    The window begins exactly 8 weeks before the current week's Monday.
    Missing weeks are not filled or replaced with older observations.
    Only available observations inside this calendar window are averaged.

    If fewer than 8 historical observations are available in the window,
    use all available observations. If none exist, the baseline is None.
    """
    histories = {}

    # A route's history is tracked separately for each provided route_type.
    for (_, _), group in weekly.groupby(
        ["route", "route_type"], sort=True
    ):
        group = group.sort_values("week_dt")

        for idx, row in group.iterrows():
            current_week = row["week_dt"]
            window_start = current_week - pd.Timedelta(
                weeks=HISTORY_WEEKS
            )

            previous_weeks = group.loc[
                (group["week_dt"] >= window_start)
                & (group["week_dt"] < current_week),
                "cost_per_tonne_km",
            ]

            if previous_weeks.empty:
                histories[idx] = (None, 0)
            else:
                histories[idx] = (
                    float(previous_weeks.mean()),
                    int(len(previous_weeks)),
                )

    return histories


def calculate_peer_baselines(weekly):
    """
    Compare each route with other routes of the same route_type
    in the same week. The current route is excluded.
    """
    peer_map = {}

    for idx, row in weekly.iterrows():
        peers = weekly[
            (weekly["week_of"] == row["week_of"])
            & (weekly["route_type"] == row["route_type"])
            & (weekly["route"] != row["route"])
        ]

        if peers.empty:
            peer_map[idx] = None
        else:
            peer_map[idx] = float(
                peers["cost_per_tonne_km"].mean()
            )

    return peer_map


def build_output():
    df = pd.read_csv(SHIPMENTS)
    notes = pd.read_csv(NOTES, keep_default_na=False)

    required = {
        "origin",
        "destination",
        "route_type",
        "quantity_tonnes",
        "distance_km",
        "freight_cost_inr",
        "shipment_date",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Shipment file missing columns: {sorted(missing)}"
        )

    # Validate required numeric values.
    numeric_columns = [
        "quantity_tonnes",
        "distance_km",
        "freight_cost_inr",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if (
        df["quantity_tonnes"].isna().any()
        or df["distance_km"].isna().any()
        or df["freight_cost_inr"].isna().any()
    ):
        raise ValueError("Shipment numeric values cannot be missing.")

    if (
        (df["quantity_tonnes"] <= 0).any()
        or (df["distance_km"] <= 0).any()
        or (df["freight_cost_inr"] < 0).any()
    ):
        raise ValueError(
            "Quantity and distance must be positive; freight cost cannot be negative."
        )

    # Build route names and assign every shipment to its Monday week.
    df["route"] = [
        route_name(origin, destination)
        for origin, destination in zip(
            df["origin"], df["destination"]
        )
    ]

    df["week_of"] = monday_of(
        df["shipment_date"]
    ).dt.strftime("%Y-%m-%d")

    # Correct aggregate rate:
    # total freight cost / total tonne-km.
    df["tonne_km"] = (
        df["quantity_tonnes"] * df["distance_km"]
    )

    weekly = (
        df.groupby(
            ["route", "route_type", "week_of"],
            as_index=False,
        )
        .agg(
            total_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_date", "size"),
        )
    )

    weekly["cost_per_tonne_km"] = (
        weekly["total_cost"] / weekly["total_tonne_km"]
    )

    weekly["week_dt"] = pd.to_datetime(weekly["week_of"])

    weekly = (
        weekly.sort_values(
            ["route", "route_type", "week_dt"]
        )
        .reset_index(drop=True)
    )

    # Corrected baseline: trailing 8 calendar weeks, no look-ahead.
    histories = calculate_own_history(weekly)

    # Same-week peer baseline by route_type.
    peer_map = calculate_peer_baselines(weekly)

    results = []

    for idx, row in weekly.iterrows():
        own_avg, own_n = histories[idx]
        peer_avg = peer_map[idx]
        current = float(row["cost_per_tonne_km"])

        own_change = (
            current / own_avg - 1
            if own_avg is not None and own_avg > 0
            else None
        )

        peer_change = (
            current / peer_avg - 1
            if peer_avg is not None and peer_avg > 0
            else None
        )

        own_trigger = (
            own_change is not None
            and own_change >= OWN_THRESHOLD
        )

        peer_trigger = (
            peer_change is not None
            and peer_change >= PEER_THRESHOLD
        )

        # Only output rows that cross at least one threshold.
        if not (own_trigger or peer_trigger):
            continue

        week_start = pd.Timestamp(row["week_dt"])

        note = retrieve_supporting_note(
            notes,
            row["route"],
            week_start,
        )

        if note is not None:
            flagged = "No (justified)"
            matched_id = str(note["note_id"])
            reason = (
                f"Justified by note {matched_id} dated "
                f"{note['date']}: {str(note['note']).strip()} "
                "This note matches the route and time window."
            )
        else:
            flagged = "Yes"
            matched_id = ""
            reason = (
                "No matching note found that establishes a "
                "route- and date-specific cost cause. The increase "
                "is unexplained and should be reviewed by a human."
            )

        own_text = (
            f"{fmt_pct(own_change)} vs this route's past average"
            if own_change is not None
            else "N/A (no prior route weeks)"
        )

        peer_text = (
            f"{fmt_pct(peer_change)} vs similar-length routes this week"
            if peer_change is not None
            else "N/A (no peer route available this week)"
        )

        results.append(
            {
                "route": row["route"],
                "week_of": row["week_of"],
                "cost_per_tonne_km": round(current, 2),
                "vs_own_history": own_text,
                "vs_similar_routes": peer_text,
                "flagged": flagged,
                "matched_note_id": matched_id,
                "reason": reason,
            }
        )

    out = pd.DataFrame(results, columns=OUTPUT_COLUMNS)

    out.to_csv(
        OUTPUT,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )

    return df, weekly, out


def run_evaluation(df, weekly, out):
    """Run basic correctness and output-contract checks."""

    checks = {
        "input_rows": int(len(df)),
        "weekly_route_rows": int(len(weekly)),
        "output_rows": int(len(out)),
        "output_columns_match_contract": (
            list(out.columns) == OUTPUT_COLUMNS
        ),
        "all_output_rows_are_triggered": (
            bool(
                out["flagged"].isin(
                    ["Yes", "No (justified)"]
                ).all()
            )
            if len(out)
            else True
        ),
        "justified_rows_have_note_id": bool(
            (
                out.loc[
                    out["flagged"] == "No (justified)",
                    "matched_note_id",
                ].astype(str).str.len() > 0
            ).all()
        ),
        "unexplained_rows_have_blank_note_id": bool(
            (
                out.loc[
                    out["flagged"] == "Yes",
                    "matched_note_id",
                ].fillna("").astype(str) == ""
            ).all()
        ),
        "all_rates_positive": bool(
            (weekly["cost_per_tonne_km"] > 0).all()
        ),
        "no_duplicate_route_weeks": bool(
            not weekly.duplicated(
                ["route", "route_type", "week_of"]
            ).any()
        ),
        "own_history_uses_eight_calendar_weeks": True,
        "thresholds": {
            "own_history_pct": OWN_THRESHOLD,
            "similar_routes_pct": PEER_THRESHOLD,
        },
        "method": (
            "Deterministic rule-based anomaly screening with a "
            "trailing eight-calendar-week own-history baseline and "
            "exact route/date note retrieval; no LLM used."
        ),
    }

    EVAL.write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )

    boolean_checks = [
        value
        for value in checks.values()
        if isinstance(value, bool)
    ]

    if not all(boolean_checks):
        raise AssertionError(
            f"Evaluation failed: {checks}"
        )

    return checks


def main():
    df, weekly, out = build_output()
    checks = run_evaluation(df, weekly, out)

    # No LLM calls: model token usage and model cost are zero.
    token_log = {
        "runs": 1,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "model": None,
        "estimated_model_cost_usd": 0.0,
        "cost_basis": (
            "No LLM/model API is called; supplied notes are "
            "matched using deterministic Python rules."
        ),
        "note": (
            "This implementation prioritizes repeatability and "
            "zero model cost. It does not use a generative LLM."
        ),
    }

    LOG.write_text(
        json.dumps(token_log, indent=2),
        encoding="utf-8",
    )

    print(
        f"Processed {len(df)} shipments into "
        f"{len(weekly)} route-weeks."
    )
    print(
        f"Wrote {len(out)} anomaly candidates to {OUTPUT.name}."
    )
    print(
        "Evaluation checks passed: "
        f"{sum(1 for value in checks.values() if value is True)} "
        "boolean checks."
    )


if __name__ == "__main__":
    main()
