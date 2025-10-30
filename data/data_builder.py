import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from data_loader import (
    load_ohlcv,
    get_available_coins,
    load_index_constituents,
)

LOOKBACK = 60          # 60 calendar days (crypto trades 7/7)
MAX_INTERP_GAP = 5     # up to 5 days of repair
FFILL_ONLY_GAP = 1     # pure ffill for single missing day


# -----------------------------------------------------------------------------
# Gap-filling helpers
# -----------------------------------------------------------------------------

def _find_nan_runs(isnan_arr):
    """
    Find contiguous NaN runs in a boolean array (True = NaN).
    Returns a list of (start_idx, end_idx) inclusive.
    """
    runs = []
    n = len(isnan_arr)
    i = 0
    while i < n:
        if isnan_arr[i]:
            start = i
            while i + 1 < n and isnan_arr[i + 1]:
                i += 1
            end = i
            runs.append((start, end))
        i += 1
    return runs


def _linear_interp_segment(values, start, end):
    """
    Linearly interpolate values[start:end+1] IN PLACE.
    Assumes values[start-1] and values[end+1] are finite and exist.
    """
    left_val = values[start - 1]
    right_val = values[end + 1]
    gap_len = (end - start + 1)
    # interpolate over gap positions
    for k in range(gap_len):
        alpha = (k + 1) / (gap_len + 1)  # goes 1/(gap+1), 2/(gap+1), ..., gap/(gap+1)
        values[start + k] = (1 - alpha) * left_val + alpha * right_val


def _fill_short_gaps_price(series, max_interp=MAX_INTERP_GAP, ffill_only=FFILL_ONLY_GAP):
    """
    Fill missing runs in a single asset's price-like series (close, high, low):
      - gap_len == 1: fill with last known value if available,
                      otherwise use next known
      - 2 <= gap_len <= max_interp: linear interpolation between
        the value before the gap and the value after the gap (if both exist)
      - gap_len > max_interp: leave NaN
    We do NOT fabricate data before the first ever observation or after the last.
    """
    s = series.copy()
    vals = s.values.astype(float)
    isnan_arr = np.isnan(vals)
    runs = _find_nan_runs(isnan_arr)

    for (start, end) in runs:
        gap_len = end - start + 1

        # Case 1: single-day hole
        if gap_len == ffill_only:
            # Try previous known value first
            if start - 1 >= 0 and np.isfinite(vals[start - 1]):
                vals[start:end+1] = vals[start - 1]
            # Else try next known value
            elif end + 1 < len(vals) and np.isfinite(vals[end + 1]):
                vals[start:end+1] = vals[end + 1]
            # Else we can't fill -> leave NaN

        # Case 2: small multi-day gap we can interpolate
        elif 2 <= gap_len <= max_interp:
            # Need valid neighbors on BOTH sides
            left_ok = (start - 1 >= 0 and np.isfinite(vals[start - 1]))
            right_ok = (end + 1 < len(vals) and np.isfinite(vals[end + 1]))
            if left_ok and right_ok:
                _linear_interp_segment(vals, start, end)
            # If we don't have both sides, we leave NaN.

        # Case 3: > max_interp -> leave NaN
        else:
            pass

    return pd.Series(vals, index=s.index)


def _fill_short_gaps_volume(series, max_interp=MAX_INTERP_GAP, ffill_only=FFILL_ONLY_GAP):
    """
    Volume is trickier. We'll work in log1p(volume):
      - gap_len == 1: forward fill with last known log-volume if available,
                      else backward fill with next known
      - 2 <= gap_len <= max_interp: linear interpolation in log1p space if both
        neighbors exist; if only left neighbor exists (gap hits end), carry it
        forward; if only right exists (gap at start), carry it backward.
      - gap_len > max_interp: leave NaN

    After filling in log space, we expm1 back. Negative results are clipped to 0.
    """
    s = series.copy()
    vals_raw = s.values.astype(float)

    # We'll work on log1p, but keep NaN for missing
    with np.errstate(invalid='ignore', divide='ignore'):
        logv = np.log1p(vals_raw)
    # logv may be -inf if vals_raw < -1 but that shouldn't happen for volume.

    isnan_arr = np.isnan(logv)
    runs = _find_nan_runs(isnan_arr)
    for (start, end) in runs:
        gap_len = end - start + 1

        left_idx = start - 1
        right_idx = end + 1
        left_ok = (left_idx >= 0 and np.isfinite(logv[left_idx]))
        right_ok = (right_idx < len(logv) and np.isfinite(logv[right_idx]))

        if gap_len == ffill_only:
            if left_ok:
                logv[start:end+1] = logv[left_idx]
            elif right_ok:
                logv[start:end+1] = logv[right_idx]
            # else leave NaN

        elif 2 <= gap_len <= max_interp:
            if left_ok and right_ok:
                # Linear interpolation in log space
                left_val = logv[left_idx]
                right_val = logv[right_idx]
                for k in range(gap_len):
                    alpha = (k + 1) / (gap_len + 1)
                    logv[start + k] = (1 - alpha) * left_val + alpha * right_val
            elif left_ok and not right_ok:
                # gap runs off the RIGHT edge (no right neighbor), carry forward left
                logv[start:end+1] = logv[left_idx]
            elif right_ok and not left_ok:
                # gap runs off the LEFT edge (no left neighbor), carry backward right
                logv[start:end+1] = logv[right_idx]
            else:
                # no neighbors -> leave NaN
                pass

        else:
            # gap_len > max_interp => leave NaN
            pass

    # Back-transform
    with np.errstate(invalid='ignore', over='ignore'):
        filled_vol = np.expm1(logv)
    # Clip negative small float noise
    filled_vol[filled_vol < 0] = 0.0

    return pd.Series(filled_vol, index=s.index)


def fill_gaps_panel(df_prices_like, kind="price"):
    """
    Apply the short-gap fill rules column by column to a wide panel DataFrame.

    kind="price"  -> _fill_short_gaps_price
    kind="volume" -> _fill_short_gaps_volume
    """
    out_cols = {}
    for col in df_prices_like.columns:
        series = df_prices_like[col]
        if kind == "price":
            out_cols[col] = _fill_short_gaps_price(series)
        elif kind == "volume":
            out_cols[col] = _fill_short_gaps_volume(series)
        else:
            raise ValueError("kind must be 'price' or 'volume'")
    return pd.DataFrame(out_cols, index=df_prices_like.index)[df_prices_like.columns]


# -----------------------------------------------------------------------------
# Panel-building, membership, eligibility
# -----------------------------------------------------------------------------

def build_panels(start_date: str, end_date: str):
    """
    1. Load OHLCV for all available coins
    2. Pivot to wide daily panels: close, high, low, volume
    3. Reindex to full daily calendar
    4. Repair short gaps:
       - price-like (close/high/low): ffill 1-day, interp <=5-day
       - volume: same idea but in log space
    """
    coins = get_available_coins(None)
    raw = load_ohlcv(coins=coins, start_date=start_date, end_date=end_date)

    # Ensure timestamp is tz-naive datetime and sorted
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    # Some datasets might already be tz-naive; handle gracefully:
    if raw["timestamp"].dt.tz is not None:
        raw["timestamp"] = raw["timestamp"].dt.tz_convert(None)
    raw = raw.sort_values(["timestamp", "coin_id"])

    def _pivot(col):
        return (
            raw.pivot(index="timestamp", columns="coin_id", values=col)
               .sort_index()
        )

    close = _pivot("close")
    high  = _pivot("high")
    low   = _pivot("low")
    vol   = _pivot("volume")

    # Reindex each panel to full daily freq (7/7 crypto calendar)
    full_idx = pd.date_range(close.index.min(), close.index.max(), freq="D")
    close = close.reindex(full_idx)
    high  = high.reindex(full_idx)
    low   = low.reindex(full_idx)
    vol   = vol.reindex(full_idx)

    # Fill small gaps per asset
    close = fill_gaps_panel(close, kind="price")
    high  = fill_gaps_panel(high,  kind="price")
    low   = fill_gaps_panel(low,   kind="price")
    vol   = fill_gaps_panel(vol,   kind="volume")

    return close, high, low, vol


def build_monthly_membership(close: pd.DataFrame,
                             start_date: str,
                             end_date: str) -> pd.DataFrame:
    """
    membership_df[day, coin] = True if coin is in the index for that whole month.

    We call load_index_constituents(period_date) for the first day of each month.
    All days of that same (year, month) share those constituents.
    """
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    all_days = close.index[(close.index >= start_ts) & (close.index <= end_ts)]
    memberships = pd.DataFrame(False, index=all_days, columns=close.columns)

    # Iterate through each calendar month in range
    cur_month = start_ts.replace(day=1)
    last_month = end_ts.replace(day=1)

    while cur_month <= last_month:
        period_date_str = cur_month.strftime("%Y-%m-%d")
        members = load_index_constituents(period_date_str)  # list[str]

        month_mask = (all_days.year == cur_month.year) & (all_days.month == cur_month.month)
        month_days = all_days[month_mask]

        # Only mark members that we actually have columns for
        valid_members = [m for m in members if m in memberships.columns]
        memberships.loc[month_days, valid_members] = True

        cur_month = (cur_month + relativedelta(months=1)).replace(day=1)

    return memberships


def apply_cold_start(memberships: pd.DataFrame,
                     close: pd.DataFrame,
                     lookback: int = LOOKBACK) -> pd.DataFrame:
    """
    An asset is eligible on day t if:
      - It's in the membership that day,
      - We have at least `lookback` consecutive non-NaN closes ending at t.

    NOTE: Because we interpolated/filled only small gaps (<=5 days),
    long gaps still remain NaN, which will break eligibility and
    force the model to treat that asset as not tradable until it
    re-establishes 60 clean days.
    """
    eligible = memberships.copy()

    for coin in close.columns:
        # 1 if we consider we have a valid close for that day, else 0
        valid_close_flag = close[coin].notna().astype(int)

        # rolling sum of last LOOKBACK days
        hist_ok = valid_close_flag.rolling(
            window=lookback,
            min_periods=lookback
        ).sum()

        # hist_ok == lookback => we had continuous valid data for the last 60 days
        can_trade = (hist_ok == lookback)

        sub = can_trade.reindex(eligible.index)
        eligible[coin] = eligible[coin] & sub

    return eligible  # boolean DataFrame [day x coin]


# -----------------------------------------------------------------------------
# Daily frame builder (state tensors + next returns)
# -----------------------------------------------------------------------------

def build_daily_frames(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    vol: pd.DataFrame,
    eligible: pd.DataFrame,
    lookback: int = LOOKBACK,
):
    """
    Produce, for each day t (from lookback-1 to len-2):
      obs_tensors[t]: np.ndarray [A_t, 4, lookback]
                       features in order:
                         0: close_norm
                         1: high_norm
                         2: low_norm
                         3: volume_norm
      next_returns[t]: np.ndarray [A_t]
                       forward simple return (close[t+1]/close[t]-1)
      coin_lists[t]  : list[str] of coins in row order of obs_tensors[t]

    Notes:
    - We normalize prices by the last close in the window (close at day t).
    - Volume is log1p, z-scored within the 60-day slice, then clipped [-5,5].
    - We do NOT include w_{t-1} here; that belongs to the environment.
    - We allow NaN next-day returns (delist/vanish); env will handle forced liquidation.
    """

    obs_tensors = {}
    next_returns = {}
    coin_lists = {}

    n_days = len(close)

    # iterate up to n_days-2 so we have t+1 for forward return
    for t_idx in range(lookback - 1, n_days - 1):
        t = close.index[t_idx]
        t_next_idx = t_idx + 1

        todays_elig = eligible.iloc[t_idx]  # row of booleans
        tradable_coins = [c for c, ok in todays_elig.items() if ok]

        if len(tradable_coins) == 0:
            continue

        X_list = []
        R_next_list = []

        sl = slice(t_idx - lookback + 1, t_idx + 1)

        for coin in tradable_coins:
            C = close[coin].iloc[sl].values.astype(float)
            H = high[coin].iloc[sl].values.astype(float)
            L_ = low[coin].iloc[sl].values.astype(float)
            V = vol[coin].iloc[sl].values.astype(float)

            # Safety: if any NaNs after fill/eligibility, skip this coin
            if (
                np.isnan(C).any()
                or np.isnan(H).any()
                or np.isnan(L_).any()
                or np.isnan(V).any()
            ):
                continue

            c_ref = C[-1]
            if (not np.isfinite(c_ref)) or c_ref == 0.0:
                continue

            # --- normalize prices by last close in the window ---
            Cn = C / c_ref
            Hn = H / c_ref
            Ln = L_ / c_ref

            # --- normalize volume within the window ---
            Vlog = np.log1p(V)
            v_mean = Vlog.mean()
            v_std = Vlog.std()
            if (not np.isfinite(v_std)) or v_std < 1e-8:
                v_std = 1.0
            Vn = (Vlog - v_mean) / v_std
            Vn = np.clip(Vn, -5, 5)

            # stack features -> [4, lookback]
            X_coin = np.stack([Cn, Hn, Ln, Vn], axis=0).astype("float32")
            X_list.append(X_coin)

            # --- compute forward return for reward ---
            c_t = close[coin].iloc[t_idx]
            c_tp1 = close[coin].iloc[t_next_idx]
            if np.isfinite(c_t) and np.isfinite(c_tp1) and c_t != 0.0:
                r_next = (c_tp1 / c_t) - 1.0
            else:
                r_next = np.nan

            R_next_list.append(r_next)

        if len(X_list) == 0:
            # It's possible all candidates got filtered out on sanity checks
            continue

        X_t = np.stack(X_list, axis=0).astype("float32")       # [A_t, 4, lookback]
        R_next_arr = np.array(R_next_list, dtype="float32")     # [A_t]

        obs_tensors[t] = X_t
        next_returns[t] = R_next_arr
        coin_lists[t] = tradable_coins

    return obs_tensors, next_returns, coin_lists


# -----------------------------------------------------------------------------
# Main (example run / sanity check)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage:
    #
    # 1. Define a small range for testing (extend later).
    # 2. Build panels (OHLCV wide, daily, gap-repaired).
    # 3. Build membership mask using index constituents.
    # 4. Apply cold-start eligibility (60-day rolling window).
    # 5. Build per-day tensors and forward returns.
    # 6. Print some diagnostics.
    from dotenv import load_dotenv
    load_dotenv()

    start_date = "2021-01-01"
    end_date = "2021-04-30"

    print(">>> Loading and preparing OHLCV panels...")
    close_df, high_df, low_df, vol_df = build_panels(start_date, end_date)
    print("close_df shape:", close_df.shape)
    print("high_df shape:", high_df.shape)
    print("low_df shape :", low_df.shape)
    print("vol_df shape :", vol_df.shape)

    print("\n>>> Building monthly membership mask...")
    membership_df = build_monthly_membership(close_df, start_date, end_date)
    print("membership_df shape:", membership_df.shape)

    print("\n>>> Applying 60-day cold-start eligibility...")
    eligible_df = apply_cold_start(membership_df, close_df, lookback=LOOKBACK)
    print("eligible_df shape:", eligible_df.shape)

    print("\n>>> Building daily tensors and next-day returns...")
    obs_tensors, fwd_returns, coin_lists = build_daily_frames(
        close_df, high_df, low_df, vol_df, eligible_df, lookback=LOOKBACK
    )

    all_days = sorted(obs_tensors.keys())
    print("Number of days with tensors:", len(all_days))

    if len(all_days) > 0:
        t_sample = all_days[-1]  # last available day
        X_sample = obs_tensors[t_sample]
        R_sample = fwd_returns[t_sample]
        coins_sample = coin_lists[t_sample]

        print(f"\nSample day: {t_sample}")
        print(" - obs_tensors[t].shape:", X_sample.shape)    # [A_t, 4, LOOKBACK]
        print(" - fwd_returns[t].shape:", R_sample.shape)    # [A_t]
        print(" - len(coin_lists[t]):", len(coins_sample))
        print(" - first few coins:", coins_sample[:5])

        # Inspect first asset's feature channels
        print(" - first asset stats:")
        print("     close_norm min/max:",
              float(X_sample[0, 0].min()), float(X_sample[0, 0].max()))
        print("     vol_norm   min/max:",
              float(X_sample[0, 3].min()), float(X_sample[0, 3].max()))

    print("\nDone.")
