"""Dataset loading and feature matrix extraction for ML training and evaluation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LoadedSplit:
    name: str
    feature_names: list[str]
    X: np.ndarray
    y: np.ndarray
    net_returns_bps: np.ndarray
    assets: list[str]
    decision_times: list[str]
    sample_metadata: list[dict[str, Any]]


def load_dataset_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found in {dataset_dir}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split(
    dataset_dir: str | Path,
    split_name: str,
    *,
    target_asset: str | None = None,
) -> LoadedSplit:
    """Load a dataset split (train, validation, test) into feature matrix and labels."""
    dataset_dir = Path(dataset_dir)
    split_file = dataset_dir / f"{split_name}.jsonl"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    manifest = load_dataset_manifest(dataset_dir)
    schema_version = manifest.get("feature_schema_version", "candle-market-features-v1")

    records: list[dict[str, Any]] = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if target_asset is not None:
        records = [r for r in records if r.get("asset") == target_asset]

    if not records:
        raise ValueError(f"No samples found in split '{split_name}' (filter asset={target_asset})")

    first_features = records[0]["feature_vector"]["features"]
    feature_names = [f["name"] for f in first_features]

    X_list: list[list[float]] = []
    y_list: list[int] = []
    returns_list: list[float] = []
    assets_list: list[str] = []
    times_list: list[str] = []
    metadata_list: list[dict[str, Any]] = []

    for r in records:
        f_vec = r["feature_vector"]
        f_map = {f["name"]: f["value"] for f in f_vec["features"]}

        row = []
        for name in feature_names:
            val = f_map.get(name)
            if val is None or not np.isfinite(val):
                raise ValueError(
                    f"Sample at {r.get('decision_time')} has missing or non-finite feature '{name}'"
                )
            row.append(float(val))
        X_list.append(row)

        lbl = r["label"]
        direction = lbl.get("label")
        if direction is None:
            raise ValueError(f"Sample at {r.get('decision_time')} has null label")
        y_list.append(int(direction))

        net_ret = lbl.get("net_return_bps")
        returns_list.append(float(net_ret) if net_ret is not None else 0.0)

        assets_list.append(r["asset"])
        times_list.append(r["decision_time"])
        metadata_list.append(
            {
                "asset": r["asset"],
                "decision_time": r["decision_time"],
                "label": int(direction),
                "barrier": lbl.get("barrier"),
                "net_return_bps": net_ret,
                "gross_return_bps": lbl.get("gross_return_bps"),
                "holding_bars": lbl.get("holding_bars"),
            }
        )

    return LoadedSplit(
        name=split_name,
        feature_names=feature_names,
        X=np.array(X_list, dtype=np.float64),
        y=np.array(y_list, dtype=np.int64),
        net_returns_bps=np.array(returns_list, dtype=np.float64),
        assets=assets_list,
        decision_times=times_list,
        sample_metadata=metadata_list,
    )
