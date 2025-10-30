import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# We assume the finalized tensor-building logic (gap repair, eligibility, etc.)
# lives in tensor_builder.py with the following imports:
from data_builder import (
    LOOKBACK,                 # should be 60
    build_panels,             # (start_date, end_date) -> close_df, high_df, low_df, vol_df
    build_monthly_membership, # (close_df, start_date, end_date) -> membership_df
    apply_cold_start,         # (membership_df, close_df, lookback) -> eligible_df
    build_daily_frames,       # (close_df, high_df, low_df, vol_df, eligible_df, lookback)
)

###############################################################################
# Configuration
###############################################################################

RAW_START_DATE = "2018-07-01"   # we load data from here so we have warmup context
DEV_START_DATE = "2018-09-01"   # first day we actually allow decisions
DEV_END_DATE   = "2023-12-31"

TEST_START_DATE = "2024-01-01"
TEST_END_DATE   = "2025-10-31"

# Validation windows: ~20-day slices capturing distinct regimes
VALIDATION_WINDOWS = [
    {"name": "val_2018_crash", "start": "2018-11-15", "end": "2018-12-04"},
    {"name": "val_covid",      "start": "2020-03-01", "end": "2020-03-20"},
    {"name": "val_bull",       "start": "2021-01-15", "end": "2021-02-03"},
    {"name": "val_bear",       "start": "2022-06-01", "end": "2022-06-20"},
    {"name": "val_chop",       "start": "2023-10-01", "end": "2023-10-20"},
]

# Environment / market design constants to embed in metadata
TURNOVER_CAP_L1 = 0.30
LOOKBACK_DAYS = 60
MAX_INTERP_GAP_DAYS = 5
FULLY_INVESTED = True
LONG_ONLY = True
INCLUDE_CASH_ASSET = False

DATASET_VERSION = "dataset_v1"


###############################################################################
# Helpers
###############################################################################

def date_str(dt):
    """Format Timestamp/date-like as YYYY-MM-DD string."""
    return pd.Timestamp(dt).strftime("%Y-%m-%d")

def in_window(ts, start, end):
    """Check if timestamp ts is in [start, end] inclusive (date strings)."""
    return (ts >= pd.Timestamp(start)) and (ts <= pd.Timestamp(end))

def assign_split_tag(ts):
    """
    Given a timestamp `ts` in the Dev range, assign either:
    - "val_window_<name>"
    - "train_core"
    For timestamps in Test, we'll tag separately as "test".
    """
    for win in VALIDATION_WINDOWS:
        if in_window(ts, win["start"], win["end"]):
            return f'val_window_{win["name"]}'
    return "train_core"


def build_split_indices(obs_tensors, fwd_returns, coin_lists):
    """
    Convert the dict outputs from build_daily_frames(...) into a
    sorted list of timestamps and aligned per-day structures.

    Returns:
        all_days: sorted list of timestamps available
        rows: list of dict {"date": <str>, "split_tag": <str>}
        obs_by_day: dict[timestamp] -> np.ndarray [A_t, 4, LOOKBACK]
        fwdret_by_day: dict[timestamp] -> np.ndarray [A_t]
        assets_by_day: dict[timestamp] -> list[str]
    """
    all_days = sorted(obs_tensors.keys())
    obs_by_day = {}
    fwdret_by_day = {}
    assets_by_day = {}
    for t in all_days:
        obs_by_day[t] = obs_tensors[t]
        fwdret_by_day[t] = fwd_returns[t]
        assets_by_day[t] = coin_lists[t]
    return all_days, obs_by_day, fwdret_by_day, assets_by_day


def export_index_parquet(index_rows, out_path):
    """
    index_rows: list of {"date": "YYYY-MM-DD", "split_tag": "..."}
    Saves a parquet with columns ["date", "split_tag"].
    """
    df = pd.DataFrame(index_rows)
    # enforce stable ordering by date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(out_path, index=False)


def export_asset_lists(assets_by_day, out_path):
    """
    assets_by_day: dict[timestamp] -> list[str]
    Writes JSONL. Each line:
      {"date": "YYYY-MM-DD", "assets": [...]}
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for t, assets in sorted(assets_by_day.items()):
            rec = {
                "date": date_str(t),
                "assets": assets,
            }
            f.write(json.dumps(rec) + "\n")


def export_npz_tensors(obs_by_day, out_path):
    """
    obs_by_day: dict[timestamp] -> np.ndarray [A_t, 4, LOOKBACK]
    We'll produce an .npz with keys like "t_YYYY-MM-DD".
    """
    # Build dict for savez_compressed
    npz_dict = {}
    for t, arr in obs_by_day.items():
        key = "t_" + date_str(t)
        npz_dict[key] = arr
    np.savez_compressed(out_path, **npz_dict)


def export_npz_fwd_returns(fwdret_by_day, out_path):
    """
    fwdret_by_day: dict[timestamp] -> np.ndarray [A_t]
    We'll produce an .npz with keys like "t_YYYY-MM-DD".
    """
    npz_dict = {}
    for t, arr in fwdret_by_day.items():
        key = "t_" + date_str(t)
        npz_dict[key] = arr
    np.savez_compressed(out_path, **npz_dict)


###############################################################################
# Main export logic
###############################################################################

def main():
    os.makedirs(DATASET_VERSION, exist_ok=True)

    # 1. Build cleaned OHLCV panels with repaired gaps.
    close_df, high_df, low_df, vol_df = build_panels(
        RAW_START_DATE,
        TEST_END_DATE,
    )

    # 2. Build monthly membership mask across full range, then eligibility mask.
    membership_df = build_monthly_membership(
        close_df,
        RAW_START_DATE,
        TEST_END_DATE,
    )
    eligible_df = apply_cold_start(
        membership_df,
        close_df,
        lookback=LOOKBACK,
    )

    # 3. Build per-day tensors and fwd returns for the entire horizon.
    #    These dicts will contain entries for any t that satisfies LOOKBACK,
    #    up to TEST_END_DATE-1 (because we need t+1 for forward return).
    obs_tensors, fwd_returns, coin_lists = build_daily_frames(
        close_df,
        high_df,
        low_df,
        vol_df,
        eligible_df,
        lookback=LOOKBACK,
    )

    # 4. Restructure them into aligned dicts and sorted day list
    all_days, obs_by_day, fwdret_by_day, assets_by_day = build_split_indices(
        obs_tensors,
        fwd_returns,
        coin_lists,
    )

    # 5. Split Dev vs Test by date
    dev_days = [
        d for d in all_days
        if (d >= pd.Timestamp(DEV_START_DATE)) and (d <= pd.Timestamp(DEV_END_DATE))
    ]
    test_days = [
        d for d in all_days
        if (d >= pd.Timestamp(TEST_START_DATE)) and (d <= pd.Timestamp(TEST_END_DATE))
    ]

    # 6. Build index rows with split tags
    dev_rows = []
    for d in dev_days:
        dev_rows.append({
            "date": date_str(d),
            "split_tag": assign_split_tag(d),  # "train_core" or "val_window_*"
        })

    test_rows = []
    for d in test_days:
        test_rows.append({
            "date": date_str(d),
            "split_tag": "test",
        })

    # 7. Slice per-day dicts down to Dev-only/Test-only
    dev_obs_by_day = {d: obs_by_day[d] for d in dev_days}
    dev_fwdret_by_day = {d: fwdret_by_day[d] for d in dev_days}
    dev_assets_by_day = {d: assets_by_day[d] for d in dev_days}

    test_obs_by_day = {d: obs_by_day[d] for d in test_days}
    test_fwdret_by_day = {d: fwdret_by_day[d] for d in test_days}
    test_assets_by_day = {d: assets_by_day[d] for d in test_days}

    # 8. Write parquet / npz / jsonl outputs
    export_index_parquet(
        dev_rows,
        os.path.join(DATASET_VERSION, "dev_index.parquet"),
    )
    export_index_parquet(
        test_rows,
        os.path.join(DATASET_VERSION, "test_index.parquet"),
    )

    export_npz_tensors(
        dev_obs_by_day,
        os.path.join(DATASET_VERSION, "dev_obs_tensors.npz"),
    )
    export_npz_tensors(
        test_obs_by_day,
        os.path.join(DATASET_VERSION, "test_obs_tensors.npz"),
    )

    export_asset_lists(
        dev_assets_by_day,
        os.path.join(DATASET_VERSION, "dev_asset_lists.jsonl"),
    )
    export_asset_lists(
        test_assets_by_day,
        os.path.join(DATASET_VERSION, "test_asset_lists.jsonl"),
    )

    export_npz_fwd_returns(
        dev_fwdret_by_day,
        os.path.join(DATASET_VERSION, "dev_fwd_returns.npz"),
    )
    export_npz_fwd_returns(
        test_fwdret_by_day,
        os.path.join(DATASET_VERSION, "test_fwd_returns.npz"),
    )

    # 9. Write metadata.json
    metadata = {
        "dataset_version": DATASET_VERSION,
        "raw_start_date": RAW_START_DATE,
        "dev_start_date": DEV_START_DATE,
        "dev_end_date": DEV_END_DATE,
        "test_start_date": TEST_START_DATE,
        "test_end_date": TEST_END_DATE,
        "lookback_days": LOOKBACK_DAYS,
        "max_interp_gap_days": MAX_INTERP_GAP_DAYS,
        "min_history_for_eligibility_days": LOOKBACK_DAYS,
        "turnover_cap_l1": TURNOVER_CAP_L1,
        "fully_invested": FULLY_INVESTED,
        "long_only": LONG_ONLY,
        "include_cash_asset": INCLUDE_CASH_ASSET,
        "validation_windows": VALIDATION_WINDOWS,
        "notes": (
            "This dataset enforces monthly membership, 60-day cold start, "
            "gap repair up to 5 days, no cash asset, long-only fully "
            "invested portfolios, and discontiguous regime validation windows. "
            "Forward returns are provided only for reward calculation; they "
            "are not part of the observation tensor."
        ),
        # Citations you will surface in the paper / thesis:
        "citations": [
            "jiang2016drlt",
            "jiang2017eIIE",
            "lucarelli2020dqlcrypto",
        ],
    }

    with open(os.path.join(DATASET_VERSION, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Export complete in directory: {DATASET_VERSION}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
