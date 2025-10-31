import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any


class ExportedDataset:
    """
    Lightweight container for a loaded RL dataset split.

    Attributes
    ----------
    split : str
        "dev" or "test".
    index_df : pd.DataFrame
        Columns: ["date", "split_tag"], sorted ascending by date.
        date is pandas.Timestamp (tz-naive, daily).
    obs_tensors : Dict[str, np.ndarray]
        Mapping "YYYY-MM-DD" -> observation tensor [A_t, 4, 60] (float32).
    asset_lists : Dict[str, list]
        Mapping "YYYY-MM-DD" -> list[str] asset IDs aligned with obs_tensors rows.
    fwd_returns : Dict[str, np.ndarray]
        Mapping "YYYY-MM-DD" -> forward returns [A_t] (float32), aligned with obs_tensors rows.
    metadata : Dict[str, Any]
        Parsed metadata.json for the dataset root (same for dev/test).
    """

    def __init__(
        self,
        split: str,
        index_df: pd.DataFrame,
        obs_tensors: Dict[str, np.ndarray],
        asset_lists: Dict[str, list],
        fwd_returns: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
    ):
        self.split = split
        self.index_df = index_df
        self.obs_tensors = obs_tensors
        self.asset_lists = asset_lists
        self.fwd_returns = fwd_returns
        self.metadata = metadata

    def dates(self):
        """Return list of all date strings (YYYY-MM-DD) in this split, in order."""
        return [d.strftime("%Y-%m-%d") for d in self.index_df["date"].tolist()]

    def get_day(self, date_str: str):
        """
        Convenience accessor to grab everything for a given YYYY-MM-DD.

        Returns
        -------
        {
            "obs_tensor": np.ndarray [A_t, 4, 60],
            "assets": list[str] length A_t,
            "fwd_returns": np.ndarray [A_t],
            "split_tag": str,
            "timestamp": pd.Timestamp
        }
        Raises KeyError if the date is not present.
        """
        # locate metadata row
        ts = pd.Timestamp(date_str)
        row = self.index_df.loc[self.index_df["date"] == ts]
        if row.empty:
            raise KeyError(f"{date_str} not found in index_df for split {self.split}")
        split_tag = row["split_tag"].iloc[0]

        return {
            "obs_tensor": self.obs_tensors[date_str],
            "assets": self.asset_lists[date_str],
            "fwd_returns": self.fwd_returns[date_str],
            "split_tag": split_tag,
            "timestamp": ts,
        }


def _load_index_parquet(path_parquet: str) -> pd.DataFrame:
    """
    Read dev_index.parquet or test_index.parquet and return a clean DataFrame
    with columns ["date", "split_tag"] and date as Timestamp.
    """
    df = pd.read_parquet(path_parquet)
    # Normalize types
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "split_tag"]]


def _load_asset_lists_jsonl(path_jsonl: str) -> Dict[str, list]:
    """
    Read dev_asset_lists.jsonl or test_asset_lists.jsonl into a dict:
    { "YYYY-MM-DD": ["BTC", "ETH", ...], ... }
    """
    out = {}
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # rec["date"] is already "YYYY-MM-DD"
            out[rec["date"]] = rec["assets"]
    return out


def _load_npz_dict(path_npz: str) -> Dict[str, np.ndarray]:
    """
    Read an .npz where keys look like "t_YYYY-MM-DD" and return
    a dict { "YYYY-MM-DD": np.ndarray(...) }.

    This is used for:
    - dev_obs_tensors.npz / test_obs_tensors.npz
    - dev_fwd_returns.npz / test_fwd_returns.npz
    """
    data = np.load(path_npz, allow_pickle=False)
    out = {}
    for key in data.files:
        # Expect key format "t_YYYY-MM-DD"
        if not key.startswith("t_"):
            raise ValueError(f"Unexpected key in {path_npz}: {key}")
        date_str = key[2:]  # strip leading "t_"
        out[date_str] = data[key]
    return out


def _load_metadata(path_json: str) -> Dict[str, Any]:
    """
    Load metadata.json from dataset root.
    """
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)


def load_exported_dataset(dataset_dir: str, split: str) -> ExportedDataset:
    """
    Load a pre-exported dataset split ("dev" or "test") from dataset_dir.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset_vN directory we exported, e.g. "dataset_v1".
        This directory is expected to contain:
            metadata.json
            dev_index.parquet / test_index.parquet
            dev_obs_tensors.npz / test_obs_tensors.npz
            dev_asset_lists.jsonl / test_asset_lists.jsonl
            dev_fwd_returns.npz / test_fwd_returns.npz
    split : str
        "dev" or "test".

    Returns
    -------
    ExportedDataset
        An object with:
        - index_df (pd.DataFrame)
        - obs_tensors (dict[str -> np.ndarray])
        - asset_lists (dict[str -> list[str]])
        - fwd_returns (dict[str -> np.ndarray])
        - metadata (dict)
    """

    split = split.lower().strip()
    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test'")

    # figure out file names for this split
    index_path = os.path.join(dataset_dir, f"{split}_index.parquet")
    obs_path = os.path.join(dataset_dir, f"{split}_obs_tensors.npz")
    assets_path = os.path.join(dataset_dir, f"{split}_asset_lists.jsonl")
    fwdret_path = os.path.join(dataset_dir, f"{split}_fwd_returns.npz")
    meta_path = os.path.join(dataset_dir, "metadata.json")

    # sanity checks
    required_paths = [index_path, obs_path, assets_path, fwdret_path, meta_path]
    for p in required_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected dataset file not found: {p}")

    # load components
    index_df = _load_index_parquet(index_path)
    asset_lists = _load_asset_lists_jsonl(assets_path)
    obs_tensors = _load_npz_dict(obs_path)
    fwd_returns = _load_npz_dict(fwdret_path)
    metadata = _load_metadata(meta_path)

    # basic consistency checks
    # dates we have in the index drive everything
    date_strs_from_index = [d.strftime("%Y-%m-%d") for d in index_df["date"].tolist()]

    missing_obs = [d for d in date_strs_from_index if d not in obs_tensors]
    missing_assets = [d for d in date_strs_from_index if d not in asset_lists]
    missing_fwd = [d for d in date_strs_from_index if d not in fwd_returns]

    if missing_obs:
        raise ValueError(f"obs_tensors missing dates: {missing_obs[:5]} ...")
    if missing_assets:
        raise ValueError(f"asset_lists missing dates: {missing_assets[:5]} ...")
    if missing_fwd:
        raise ValueError(f"fwd_returns missing dates: {missing_fwd[:5]} ...")

    # we could also assert shapes line up:
    for d in date_strs_from_index:
        A_obs = obs_tensors[d].shape[0]
        A_assets = len(asset_lists[d])
        A_fwd = fwd_returns[d].shape[0]
        if not (A_obs == A_assets == A_fwd):
            raise ValueError(
                f"Mismatch for {d}: "
                f"A_obs={A_obs}, A_assets={A_assets}, A_fwd={A_fwd}"
            )

    # build and return the object
    return ExportedDataset(
        split=split,
        index_df=index_df,
        obs_tensors=obs_tensors,
        asset_lists=asset_lists,
        fwd_returns=fwd_returns,
        metadata=metadata,
    )

if __name__ == "__main__":
    # extract dataset_v1 into a certain directory and load both splits
    dev_data = load_exported_dataset("dataset_v1", split="dev")
    test_data = load_exported_dataset("dataset_v1", split="test")

    print(dev_data.metadata["dev_start_date"], "→", dev_data.metadata["dev_end_date"])
    print(test_data.metadata["test_start_date"], "→", test_data.metadata["test_end_date"])

    # get the first decision day in dev:
    first_day = dev_data.dates()[0]
    sample = dev_data.get_day(first_day)
    print("Day:", first_day)
    print("Split tag:", sample["split_tag"])
    print("Obs tensor shape:", sample["obs_tensor"].shape)   # [A_t, 4, 60]
    print("Num assets:", len(sample["assets"]))
    print("Forward returns shape:", sample["fwd_returns"].shape)  # [A_t]