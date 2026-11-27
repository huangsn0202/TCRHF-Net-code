"""Load one offline-encoded T/I/C vector per news item (paper sections 3.7, 4.1)."""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "feature" / "goosipcop"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data" / "goosipcop" / "dataset_split.xlsx"
FEATURE_DIM = 1024
VALID_SPLITS = {"train", "test"}


def _normalise_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("The split table contains a missing id.")
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text or text.lower() == "nan":
        raise ValueError("The split table contains an empty id.")
    if any(character in text for character in ("/", "\\", "\x00")) or text in (".", ".."):
        raise ValueError(f"Invalid sample id {text!r}.")
    return text


def _read_split_table(split_file: Path) -> pd.DataFrame:
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    table = pd.read_excel(split_file, engine="openpyxl", dtype=str, keep_default_na=False)
    table.columns = [str(column).strip().lower() for column in table.columns]
    if table.columns.duplicated().any():
        raise ValueError("Duplicate column names after normalization in split file.")
    missing = {"id", "label", "split"}.difference(table.columns)
    if missing:
        raise ValueError(f"Split file is missing columns: {sorted(missing)}")
    table["id"] = table["id"].map(_normalise_id)
    if table["id"].duplicated().any():
        raise ValueError("Duplicate normalized ids in split file; train/test must be disjoint.")
    table["split"] = table["split"].str.strip().str.lower()
    unknown = sorted(set(table["split"]) - VALID_SPLITS)
    if unknown:
        raise ValueError(f"Unknown split values: {unknown}")
    if set(table["split"]) != VALID_SPLITS:
        raise ValueError("Both non-empty train and test splits are required.")
    labels = pd.to_numeric(table["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Labels must be exactly 0 (non-fake) or 1 (fake).")
    table["label"] = labels.astype(int)
    if "event_id" in table.columns:
        table["event_id"] = table["event_id"].str.strip()
        if table["event_id"].eq("").any():
            raise ValueError("event_id cannot be empty when supplied.")
        crossings = table.groupby("event_id")["split"].nunique()
        overlap = crossings[crossings > 1].index.tolist()
        if overlap:
            raise ValueError(f"Event leakage across train/test, examples: {overlap[:10]}")
        audit = {"status": "verified_disjoint", "event_column": "event_id",
                 "event_count": int(table["event_id"].nunique())}
    else:
        warnings.warn(
            "Split table has no event_id column: the paper's event-disjoint split "
            "cannot be verified. Existing split assignments are preserved.",
            RuntimeWarning, stacklevel=2,
        )
        audit = {"status": "unverified_no_event_id", "event_column": None}
    table.attrs["event_split_audit"] = audit
    return table


def _load_feature(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.size != FEATURE_DIM or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"Feature {path.name} must contain {FEATURE_DIM} floating-point values; got {array.shape}/{array.dtype}.")
    array = np.asarray(array, dtype=np.float32).reshape(FEATURE_DIM)
    if not np.isfinite(array).all():
        raise ValueError(f"Feature {path.name} contains NaN or Inf values.")
    return torch.from_numpy(array)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_statistics(split_file: Path = DEFAULT_SPLIT_FILE) -> Dict[str, Dict[int, int]]:
    table = _read_split_table(Path(split_file))
    return {split: {int(label): int(count) for label, count in
                   table.loc[table["split"] == split, "label"].value_counts().items()}
            for split in sorted(VALID_SPLITS)}


class GossipCopFeatureDataset(Dataset):
    def __init__(
        self, split: str, feature_dir: Path = DEFAULT_FEATURE_DIR,
        split_file: Path = DEFAULT_SPLIT_FILE, verify_files: bool = True,
    ) -> None:
        split = split.strip().lower()
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split {split!r}.")
        self.feature_dir = Path(feature_dir)
        if not self.feature_dir.is_dir():
            raise FileNotFoundError(f"Feature directory not found: {self.feature_dir}")
        table = _read_split_table(Path(split_file))
        self.event_split_audit = table.attrs["event_split_audit"]
        subset = table.loc[table["split"] == split]
        self.records = [(str(row.id), int(row.label)) for row in subset.itertuples(index=False)]
        self.feature_paths = {}
        if verify_files:
            for sample_id, _ in self.records:
                for path in self._paths(sample_id):
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing feature: {path}")

    def _paths(self, sample_id: str) -> Tuple[Path, Path, Path]:
        return tuple(self.feature_dir / f"{sample_id}_{view}.npy" for view in ("T", "I", "C"))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_id, label = self.records[index]
        text_path, image_path, caption_path = self._paths(sample_id)
        return {"id": sample_id, "text": _load_feature(text_path),
                "image": _load_feature(image_path), "caption": _load_feature(caption_path),
                "label": torch.tensor(label, dtype=torch.long)}


def create_dataloaders(
    batch_size: int, seed: int, num_workers: int = 0,
    feature_dir: Path = DEFAULT_FEATURE_DIR, split_file: Path = DEFAULT_SPLIT_FILE,
    verify_files: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    train_dataset = GossipCopFeatureDataset("train", feature_dir, split_file, verify_files)
    test_dataset = GossipCopFeatureDataset("test", feature_dir, split_file, verify_files)
    common = {"batch_size": batch_size, "num_workers": num_workers,
              "pin_memory": torch.cuda.is_available(), "worker_init_fn": _seed_worker,
              "persistent_workers": num_workers > 0, "drop_last": False}
    # An independent evaluation generator avoids consuming the model RNG stream.
    return (
        DataLoader(train_dataset, shuffle=True, generator=torch.Generator().manual_seed(seed), **common),
        DataLoader(test_dataset, shuffle=False, generator=torch.Generator().manual_seed(seed), **common),
    )
