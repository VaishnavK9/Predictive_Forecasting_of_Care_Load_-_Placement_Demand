"""Load and clean the HHS UAC program CSV.

Handles four known data-quality issues in the source file:
1. ~450 trailing all-blank rows appended after the real data.
2. Dates stored as text (e.g. "December 21, 2025") in DESCENDING order.
3. `Children in HHS Care` (and occasionally other numeric columns) are
   comma-formatted text (e.g. "2,484") rather than numeric.
4. The series is NOT a clean daily series. The program reports on a
   Sun-Thu-ish cadence -- Fridays and Saturdays are almost never present.
   Roughly a third of calendar days are legitimately unobserved, not
   missing. We must not fabricate them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw CSV with no cleaning applied."""
    path = path or config.RAW_CSV_PATH
    return pd.read_csv(path)


def drop_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are entirely NaN (trailing blank rows in the source file)."""
    return df.dropna(how="all").copy()


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip thousands separators and coerce every configured numeric column to float."""
    df = df.copy()
    for col in config.NUMERIC_COLS:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_and_sort_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the text Date column, sort ascending, and set as DatetimeIndex."""
    df = df.copy()
    df[config.COL_DATE] = pd.to_datetime(df[config.COL_DATE], format="%B %d, %Y")
    df = df.sort_values(config.COL_DATE).reset_index(drop=True)
    df = df.set_index(config.COL_DATE)
    df.index.name = "date"
    return df


def reporting_cadence_report(index: pd.DatetimeIndex) -> pd.Series:
    """Return counts of observations by day-of-week, to document the actual cadence."""
    return index.day_name().value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ).fillna(0).astype(int)


def continuity_report(df: pd.DataFrame) -> dict:
    """Summarize date-range coverage: calendar days vs. observed rows vs. cadence."""
    start, end = df.index.min(), df.index.max()
    calendar_days = (end - start).days + 1
    observed = len(df)
    return {
        "start": start,
        "end": end,
        "calendar_days_in_range": calendar_days,
        "observed_rows": observed,
        "unobserved_calendar_days": calendar_days - observed,
        "pct_calendar_days_unobserved": round(100 * (calendar_days - observed) / calendar_days, 1),
        "cadence_by_weekday": reporting_cadence_report(df.index).to_dict(),
    }


def reindex_to_reporting_cadence(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto a custom Sun-Thu reporting-day frequency.

    Frequency-strategy decision (documented, not silent):
    Empirically (see `reporting_cadence_report`), the source reports on a
    Sun-Thu cadence: Sunday and Monday-Thursday are all well represented
    (~130-150 obs each across 720 rows), while Friday appears only twice
    and Saturday never. Standard pandas business-day frequency ('B' =
    Mon-Fri) would misalign this: it would slot in Fridays that are
    essentially never reported (silently invented NaN placeholders in the
    wrong place) while dropping genuine Sunday observations from the index
    entirely. Instead we build a `CustomBusinessDay` frequency with
    weekmask "Sun Mon Tue Wed Thu", which reproduces the true reporting
    week exactly -- no real observation is discarded, and no
    systematically-absent Friday/Saturday slot is fabricated.
    Only short internal gaps (<= MAX_GAP_INTERPOLATE consecutive reporting
    days -- e.g. a holiday closure) are linearly interpolated. Longer gaps
    are left as NaN.
    """
    df = df.copy()

    reporting_day = pd.offsets.CustomBusinessDay(weekmask=config.REPORTING_WEEKMASK)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=reporting_day)
    reindexed = df.reindex(full_index)
    reindexed.index.name = "date"

    is_missing = reindexed[config.TARGET_CARE_LOAD].isna()
    gap_id = (is_missing != is_missing.shift()).cumsum()
    gap_sizes = is_missing.groupby(gap_id).transform("sum")
    short_gap_mask = is_missing & (gap_sizes <= config.MAX_GAP_INTERPOLATE)

    for col in config.NUMERIC_COLS:
        interpolated = reindexed[col].interpolate(method="linear", limit_area="inside")
        reindexed[col] = reindexed[col].where(~short_gap_mask, interpolated)

    return reindexed


def add_net_flow_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Derived pressure signal: transfers into HHS minus discharges out."""
    df = df.copy()
    df[config.NET_FLOW_COL] = df[config.COL_TRANSFERRED] - df[config.COL_DISCHARGED]
    return df


def load_clean_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: raw -> cleaned (irregular index) -> reindexed (business-day-ish).

    Returns (clean_df, business_day_df).
    """
    raw = load_raw()
    df = drop_blank_rows(raw)
    df = clean_numeric_columns(df)
    df = parse_and_sort_dates(df)
    df = add_net_flow_pressure(df)

    bday_df = reindex_to_reporting_cadence(df)
    bday_df = add_net_flow_pressure(bday_df) if config.NET_FLOW_COL not in bday_df else bday_df

    return df, bday_df


def main() -> None:
    clean_df, bday_df = load_clean_dataset()

    print("=" * 70)
    print("CLEANED DATASET (irregular reporting-day index)")
    print("=" * 70)
    print(f"Shape: {clean_df.shape}")
    print(f"Date range: {clean_df.index.min().date()} -> {clean_df.index.max().date()}")
    print("\nDtypes:")
    print(clean_df.dtypes)
    print("\nDescribe:")
    print(clean_df.describe().round(1))

    report = continuity_report(clean_df)
    print("\n" + "=" * 70)
    print("CONTINUITY / CADENCE REPORT")
    print("=" * 70)
    for k, v in report.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 70)
    print("BUSINESS-DAY REINDEXED DATASET")
    print("=" * 70)
    print(f"Shape: {bday_df.shape}")
    print(f"Date range: {bday_df.index.min().date()} -> {bday_df.index.max().date()}")
    n_missing = bday_df[config.TARGET_CARE_LOAD].isna().sum()
    print(f"Missing (unobserved) rows for '{config.TARGET_CARE_LOAD}': {n_missing} "
          f"({100 * n_missing / len(bday_df):.1f}% of reindexed slots)")
    print("\nMissing-by-weekday (unobserved slots only):")
    missing_mask = bday_df[config.TARGET_CARE_LOAD].isna()
    print(bday_df.index[missing_mask].day_name().value_counts())

    clean_df.to_csv(config.CLEAN_CSV_PATH)
    bday_df.to_csv(config.BUSINESS_DAY_CSV_PATH)
    print(f"\nSaved cleaned dataset -> {config.CLEAN_CSV_PATH}")
    print(f"Saved business-day dataset -> {config.BUSINESS_DAY_CSV_PATH}")


if __name__ == "__main__":
    main()
