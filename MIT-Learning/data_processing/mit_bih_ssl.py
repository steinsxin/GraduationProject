from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .Filtering import ECG_Filter

try:
    import wfdb
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "wfdb is required for MIT-BIH processing. Install it with `pip install wfdb`."
    ) from exc


VALID_BEAT_SYMBOLS = {
    "N",
    "L",
    "R",
    "e",
    "j",
    "A",
    "a",
    "J",
    "S",
    "V",
    "E",
    "F",
    "/",
    "f",
    "Q",
}
NORMAL_BEAT_SYMBOLS = {"N", "L", "R", "e", "j"}


@dataclass
class PreparedSplits:
    labeled_X: np.ndarray
    labeled_y: np.ndarray
    unlabeled_X: np.ndarray
    unlabeled_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    metadata: Dict[str, object]


def _standardize_segment(segment: np.ndarray) -> np.ndarray:
    mean = float(np.mean(segment))
    std = float(np.std(segment))
    if std < 1e-8:
        return (segment - mean).astype(np.float32)
    return ((segment - mean) / std).astype(np.float32)


def _choose_channel(sig_names: Sequence[str]) -> int:
    if not sig_names:
        return 0
    for preferred in ("MLII", "II", "V5", "V1"):
        if preferred in sig_names:
            return sig_names.index(preferred)
    return 0


def list_record_ids(data_dir: Path) -> List[str]:
    pattern = re.compile(r"^\d+$")
    candidates = []
    for hea_path in data_dir.glob("*.hea"):
        record_id = hea_path.stem
        if not pattern.match(record_id):
            continue
        if not (data_dir / f"{record_id}.dat").exists():
            continue
        if not (data_dir / f"{record_id}.atr").exists():
            continue
        candidates.append(record_id)
    return sorted(candidates)


def _extract_record_segments(
    data_dir: Path,
    record_id: str,
    window_size: int,
    line_freq: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    record_path = str(data_dir / record_id)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    channel_idx = _choose_channel(record.sig_name)
    signal = record.p_signal[:, channel_idx]
    filtered_signal = ECG_Filter(signal, fs=int(record.fs), line_freq=line_freq)

    half_window = window_size // 2
    segments: List[np.ndarray] = []
    labels: List[int] = []
    record_ids: List[str] = []

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in VALID_BEAT_SYMBOLS:
            continue

        start = sample - half_window
        end = sample + half_window
        if start < 0 or end > len(filtered_signal):
            continue

        segment = _standardize_segment(filtered_signal[start:end])
        label = 0 if symbol in NORMAL_BEAT_SYMBOLS else 1
        segments.append(segment)
        labels.append(label)
        record_ids.append(record_id)

    if not segments:
        return (
            np.empty((0, window_size), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype="U8"),
        )

    return (
        np.stack(segments).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(record_ids),
    )


def build_or_load_cache(
    data_dir: str | Path,
    cache_path: str | Path,
    window_size: int,
    line_freq: float = 60.0,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_rebuild:
        cache = np.load(cache_path, allow_pickle=True)
        return cache["X"], cache["y"], cache["record_ids"]

    record_ids = list_record_ids(data_dir)
    if not record_ids:
        raise FileNotFoundError(f"No MIT-BIH records found under {data_dir}")

    all_segments: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_record_ids: List[np.ndarray] = []

    for record_id in tqdm(record_ids, desc="Preparing MIT-BIH"):
        segments, labels, segment_record_ids = _extract_record_segments(
            data_dir=data_dir,
            record_id=record_id,
            window_size=window_size,
            line_freq=line_freq,
        )
        if len(labels) == 0:
            continue
        all_segments.append(segments)
        all_labels.append(labels)
        all_record_ids.append(segment_record_ids)

    if not all_segments:
        raise RuntimeError("No usable beat segments were extracted from MIT-BIH.")

    X = np.concatenate(all_segments, axis=0)
    y = np.concatenate(all_labels, axis=0)
    sample_record_ids = np.concatenate(all_record_ids, axis=0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, record_ids=sample_record_ids)
    return X, y, sample_record_ids


def _sample_balanced_indices(
    y: np.ndarray,
    rng: np.random.Generator,
    max_per_class: int | None,
) -> np.ndarray:
    selected = []
    for label in (0, 1):
        label_indices = np.where(y == label)[0]
        if len(label_indices) == 0:
            continue
        if max_per_class is None or len(label_indices) <= max_per_class:
            picked = label_indices
        else:
            picked = rng.choice(label_indices, size=max_per_class, replace=False)
        selected.append(np.sort(picked))

    if not selected:
        return np.empty((0,), dtype=np.int64)
    return np.sort(np.concatenate(selected))


def _subset_arrays(
    X: np.ndarray,
    y: np.ndarray,
    record_ids: np.ndarray,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return X[indices], y[indices], record_ids[indices]


def _build_split_masks(
    record_ids: np.ndarray,
    train_records: Sequence[str],
    val_records: Sequence[str],
    test_records: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.isin(record_ids, np.asarray(train_records))
    val_mask = np.isin(record_ids, np.asarray(val_records))
    test_mask = np.isin(record_ids, np.asarray(test_records))
    return train_mask, val_mask, test_mask


def prepare_self_training_splits(
    data_dir: str | Path,
    cache_path: str | Path,
    window_size: int = 720,
    labeled_fraction: float = 0.1,
    val_ratio: float = 1.0 / 11.0,
    test_ratio: float = 0.15,
    max_train_per_class: int = 4000,
    max_eval_per_class: int = 1000,
    seed: int = 45,
    force_rebuild: bool = False,
) -> PreparedSplits:
    rng = np.random.default_rng(seed)
    X, y, sample_record_ids = build_or_load_cache(
        data_dir=data_dir,
        cache_path=cache_path,
        window_size=window_size,
        force_rebuild=force_rebuild,
    )

    unique_records = np.unique(sample_record_ids)
    train_records, test_records = train_test_split(
        unique_records,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )
    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
    train_records, val_records = train_test_split(
        train_records,
        test_size=adjusted_val_ratio,
        random_state=seed,
        shuffle=True,
    )

    train_mask, val_mask, test_mask = _build_split_masks(
        sample_record_ids,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
    )

    train_record_set = set(train_records.tolist())
    val_record_set = set(val_records.tolist())
    test_record_set = set(test_records.tolist())
    if train_record_set & val_record_set:
        raise RuntimeError("Train and validation records overlap.")
    if train_record_set & test_record_set:
        raise RuntimeError("Train and test records overlap.")
    if val_record_set & test_record_set:
        raise RuntimeError("Validation and test records overlap.")

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]

    train_keep = _sample_balanced_indices(y[train_indices], rng, max_train_per_class)
    val_keep = _sample_balanced_indices(y[val_indices], rng, max_eval_per_class)
    test_keep = _sample_balanced_indices(y[test_indices], rng, max_eval_per_class)

    train_X, train_y, train_record_ids = _subset_arrays(X[train_indices], y[train_indices], sample_record_ids[train_indices], train_keep)
    val_X, val_y, _ = _subset_arrays(X[val_indices], y[val_indices], sample_record_ids[val_indices], val_keep)
    test_X, test_y, _ = _subset_arrays(X[test_indices], y[test_indices], sample_record_ids[test_indices], test_keep)

    labeled_indices: List[np.ndarray] = []
    unlabeled_indices: List[np.ndarray] = []
    for label in (0, 1):
        label_indices = np.where(train_y == label)[0]
        if len(label_indices) == 0:
            continue
        labeled_count = max(1, int(len(label_indices) * labeled_fraction))
        rng.shuffle(label_indices)
        labeled_indices.append(np.sort(label_indices[:labeled_count]))
        unlabeled_indices.append(np.sort(label_indices[labeled_count:]))

    labeled_keep = np.sort(np.concatenate(labeled_indices)) if labeled_indices else np.empty((0,), dtype=np.int64)
    unlabeled_keep = np.sort(np.concatenate(unlabeled_indices)) if unlabeled_indices else np.empty((0,), dtype=np.int64)

    labeled_X, labeled_y, labeled_record_ids = _subset_arrays(train_X, train_y, train_record_ids, labeled_keep)
    unlabeled_X, unlabeled_y, unlabeled_record_ids = _subset_arrays(train_X, train_y, train_record_ids, unlabeled_keep)

    metadata = {
        "window_size": window_size,
        "seed": seed,
        "train_val_ratio": "10:1",
        "test_train_ratio": "about 1:5",
        "train_records": train_records.tolist(),
        "val_records": val_records.tolist(),
        "test_records": test_records.tolist(),
        "labeled_records": np.unique(labeled_record_ids).tolist(),
        "unlabeled_records": np.unique(unlabeled_record_ids).tolist(),
        "train_size": int(len(train_y)),
        "labeled_size": int(len(labeled_y)),
        "unlabeled_size": int(len(unlabeled_y)),
        "val_size": int(len(val_y)),
        "test_size": int(len(test_y)),
        "train_class_distribution": {
            "normal": int(np.sum(train_y == 0)),
            "abnormal": int(np.sum(train_y == 1)),
        },
        "labeled_class_distribution": {
            "normal": int(np.sum(labeled_y == 0)),
            "abnormal": int(np.sum(labeled_y == 1)),
        },
        "unlabeled_class_distribution": {
            "normal": int(np.sum(unlabeled_y == 0)),
            "abnormal": int(np.sum(unlabeled_y == 1)),
        },
        "val_class_distribution": {
            "normal": int(np.sum(val_y == 0)),
            "abnormal": int(np.sum(val_y == 1)),
        },
        "test_class_distribution": {
            "normal": int(np.sum(test_y == 0)),
            "abnormal": int(np.sum(test_y == 1)),
        },
    }

    return PreparedSplits(
        labeled_X=labeled_X,
        labeled_y=labeled_y,
        unlabeled_X=unlabeled_X,
        unlabeled_y=unlabeled_y,
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y,
        metadata=metadata,
    )