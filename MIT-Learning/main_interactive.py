from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple, Type

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data_processing.mit_bih_ssl import PreparedSplits, prepare_self_training_splits
from model.CNN import CNN
from model.LSTM import LSTM_Model


BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3
TOTAL_ROUNDS = 10
WARMUP_ROUNDS = 3
CONF_TH_HIGH = 0.95
CONF_TH_LOW = 0.05
MAX_PSEUDO_PER_CLASS = 800
MIN_NEW_SAMPLES_PER_CLASS = 32
FIXED_SEED = 45
MIN_CONF_TH_HIGH = 0.85
MAX_CONF_TH_LOW = 0.15
THRESHOLD_STEP = 0.05
LSTM_TO_CNN_MAX_F1_GAP = 0.05
MIN_LSTM_VAL_F1 = 0.68


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


class AugmentedDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        signal = self.X[idx].copy()
        target = self.y[idx]

        if self.augment:
            if np.random.rand() < 0.5:
                signal *= np.random.uniform(0.85, 1.15)
            if np.random.rand() < 0.5:
                signal += np.random.normal(0.0, 0.03, size=signal.shape)
            if np.random.rand() < 0.4:
                signal = np.roll(signal, np.random.randint(-24, 25))
            if np.random.rand() < 0.2:
                mask_len = min(60, len(signal) // 8)
                start = np.random.randint(0, len(signal) - mask_len)
                signal[start:start + mask_len] = 0.0

        return torch.from_numpy(signal).float().unsqueeze(0), torch.tensor(target).float()


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    shuffle: bool,
    augment: bool = False,
    seed_offset: int = 0,
) -> DataLoader:
    dataset = AugmentedDataset(X, y, augment=augment)
    generator = torch.Generator()
    generator.manual_seed(FIXED_SEED + seed_offset)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> Dict[str, float]:
    model.eval()
    losses = []
    probs = []
    targets = []

    with torch.no_grad():
        for signals, labels in data_loader:
            signals = signals.to(device)
            labels = labels.view(-1, 1).to(device)
            outputs = model(signals)
            if criterion is not None:
                losses.append(criterion(outputs, labels).item())
            probs.extend(outputs.cpu().numpy().ravel())
            targets.extend(labels.cpu().numpy().ravel())

    probs_array = np.asarray(probs, dtype=np.float32)
    preds = (probs_array >= 0.5).astype(np.int64)
    targets_array = np.asarray(targets, dtype=np.int64)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(targets_array, preds)),
        "f1": float(f1_score(targets_array, preds, zero_division=0)),
        "precision": float(precision_score(targets_array, preds, zero_division=0)),
        "recall": float(recall_score(targets_array, preds, zero_division=0)),
    }


def predict_probabilities(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    inputs = torch.from_numpy(X).float().unsqueeze(1)
    loader = DataLoader(TensorDataset(inputs), batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    probs = []
    with torch.no_grad():
        for (signals,) in loader:
            outputs = model(signals.to(device))
            probs.extend(outputs.cpu().numpy().ravel())

    return np.asarray(probs, dtype=np.float32)


def select_high_confidence_samples(
    probabilities: np.ndarray,
    max_per_class: int,
    min_per_class: int,
) -> Tuple[np.ndarray, np.ndarray]:
    positive_indices = np.where(probabilities >= CONF_TH_HIGH)[0]
    negative_indices = np.where(probabilities <= CONF_TH_LOW)[0]

    if len(positive_indices) < min_per_class or len(negative_indices) < min_per_class:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    positive_keep = positive_indices[np.argsort(probabilities[positive_indices])[::-1][:max_per_class]]
    negative_keep = negative_indices[np.argsort(probabilities[negative_indices])[:max_per_class]]
    return np.sort(positive_keep), np.sort(negative_keep)


def select_cross_feed_batch(
    source_probs: np.ndarray,
    peer_probs: np.ndarray,
    round_idx: int,
    max_per_class: int,
    min_per_class: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    high_th = max(CONF_TH_HIGH - THRESHOLD_STEP * (round_idx - 1), MIN_CONF_TH_HIGH)
    low_th = min(CONF_TH_LOW + THRESHOLD_STEP * (round_idx - 1), MAX_CONF_TH_LOW)
    avg_probs = (source_probs + peer_probs) / 2.0

    strict_pos = np.where((source_probs >= high_th) & (peer_probs >= 0.5))[0]
    strict_neg = np.where((source_probs <= low_th) & (peer_probs <= 0.5))[0]

    pos_candidates = strict_pos
    neg_candidates = strict_neg
    mode = "strict"

    if len(pos_candidates) < min_per_class or len(neg_candidates) < min_per_class:
        pos_candidates = np.where((source_probs >= 0.5) & (peer_probs >= 0.5))[0]
        neg_candidates = np.where((source_probs < 0.5) & (peer_probs < 0.5))[0]
        mode = "agreement"

    target_count = min(len(pos_candidates), len(neg_candidates), max_per_class)
    if target_count < min_per_class:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), {
            "high_th": high_th,
            "low_th": low_th,
            "mode": mode,
            "selected_per_class": 0,
        }

    pos_scores = avg_probs[pos_candidates]
    neg_scores = 1.0 - avg_probs[neg_candidates]
    pos_selected = pos_candidates[np.argsort(pos_scores)[::-1][:target_count]]
    neg_selected = neg_candidates[np.argsort(neg_scores)[::-1][:target_count]]

    selected_indices = np.concatenate([pos_selected, neg_selected])
    selected_labels = np.concatenate(
        [
            np.ones(len(pos_selected), dtype=np.float32),
            np.zeros(len(neg_selected), dtype=np.float32),
        ]
    )

    return selected_indices, selected_labels, {
        "high_th": high_th,
        "low_th": low_th,
        "mode": mode,
        "selected_per_class": int(target_count),
    }


def select_cnn_warmup_batch(
    cnn_probs: np.ndarray,
    round_idx: int,
    max_per_class: int,
    min_per_class: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    high_th = max(CONF_TH_HIGH - THRESHOLD_STEP * (round_idx - 1), MIN_CONF_TH_HIGH)
    low_th = min(CONF_TH_LOW + THRESHOLD_STEP * (round_idx - 1), MAX_CONF_TH_LOW)

    pos_candidates = np.where(cnn_probs >= high_th)[0]
    neg_candidates = np.where(cnn_probs <= low_th)[0]
    mode = "strict"

    target_count = min(len(pos_candidates), len(neg_candidates), max_per_class)
    if target_count < min_per_class:
        pos_candidates = np.where(cnn_probs >= 0.5)[0]
        neg_candidates = np.where(cnn_probs < 0.5)[0]
        target_count = min(len(pos_candidates), len(neg_candidates), max_per_class)
        mode = "ranked"

    if target_count < min_per_class:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), {
            "high_th": high_th,
            "low_th": low_th,
            "mode": mode,
            "selected_per_class": 0,
        }

    pos_selected = pos_candidates[np.argsort(cnn_probs[pos_candidates])[::-1][:target_count]]
    neg_selected = neg_candidates[np.argsort(cnn_probs[neg_candidates])[:target_count]]
    selected_indices = np.concatenate([pos_selected, neg_selected])
    selected_labels = np.concatenate(
        [
            np.ones(len(pos_selected), dtype=np.float32),
            np.zeros(len(neg_selected), dtype=np.float32),
        ]
    )

    return selected_indices, selected_labels, {
        "high_th": high_th,
        "low_th": low_th,
        "mode": mode,
        "selected_per_class": int(target_count),
    }


def train_single_model(
    model_class: Type[nn.Module],
    model_name: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    seed_offset: int,
) -> nn.Module:
    set_seed(FIXED_SEED + seed_offset)
    train_loader = make_loader(train_X, train_y, shuffle=True, augment=True, seed_offset=seed_offset)
    model = model_class().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    save_path = save_dir / f"{model_name}.pth"
    best_f1 = -1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.view(-1, 1).to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend((outputs >= 0.5).int().cpu().numpy().ravel())
            train_targets.extend(labels.int().cpu().numpy().ravel())

        train_acc = accuracy_score(train_targets, train_preds)
        train_loss = float(np.mean(train_losses))
        val_metrics = evaluate_model(model, val_loader, device, criterion)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), save_path)

        if epoch in {10, 15}:
            new_lr = {10: 5e-4, 15: 1e-4}[epoch]
            for group in optimizer.param_groups:
                group["lr"] = new_lr

        print(
            f"{model_name} Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}"
        )

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def evaluate_ensemble(
    model_cnn: nn.Module,
    model_lstm: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    cnn_weight: float = 0.5,
    lstm_weight: float = 0.5,
) -> Dict[str, float]:
    model_cnn.eval()
    model_lstm.eval()
    probs_cnn = []
    probs_lstm = []
    targets = []

    with torch.no_grad():
        for signals, labels in data_loader:
            signals = signals.to(device)
            probs_cnn.extend(model_cnn(signals).cpu().numpy().ravel())
            probs_lstm.extend(model_lstm(signals).cpu().numpy().ravel())
            targets.extend(labels.numpy().ravel())

    total_weight = cnn_weight + lstm_weight
    if total_weight <= 0:
        cnn_weight, lstm_weight = 0.5, 0.5
        total_weight = 1.0
    ensemble_probs = (cnn_weight * np.asarray(probs_cnn) + lstm_weight * np.asarray(probs_lstm)) / total_weight
    preds = (ensemble_probs >= 0.5).astype(np.int64)
    targets_array = np.asarray(targets, dtype=np.int64)

    return {
        "accuracy": float(accuracy_score(targets_array, preds)),
        "f1": float(f1_score(targets_array, preds, zero_division=0)),
        "precision": float(precision_score(targets_array, preds, zero_division=0)),
        "recall": float(recall_score(targets_array, preds, zero_division=0)),
    }


def compute_model_weights(cnn_val_f1: float, lstm_val_f1: float) -> Tuple[float, float]:
    cnn_score = max(cnn_val_f1, 1e-6)
    lstm_score = max(lstm_val_f1, 1e-6)
    total = cnn_score + lstm_score
    return cnn_score / total, lstm_score / total


def filter_selected_batch(
    selected_indices: np.ndarray,
    selected_labels: np.ndarray,
    blocked_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(selected_indices) == 0 or len(blocked_indices) == 0:
        return selected_indices, selected_labels

    keep_mask = ~np.isin(selected_indices, blocked_indices)
    return selected_indices[keep_mask], selected_labels[keep_mask]


def update_pseudo_sets(
    base_X: np.ndarray,
    base_y: np.ndarray,
    shared_pseudo_X: np.ndarray,
    shared_pseudo_y: np.ndarray,
    pseudo_X: np.ndarray,
    pseudo_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_parts = [base_X]
    label_parts = [base_y]

    if len(shared_pseudo_X) > 0:
        feature_parts.append(shared_pseudo_X)
        label_parts.append(shared_pseudo_y)

    if len(pseudo_X) > 0:
        feature_parts.append(pseudo_X)
        label_parts.append(pseudo_y)

    train_X = np.concatenate(feature_parts, axis=0)
    train_y = np.concatenate(label_parts, axis=0)

    permutation = np.random.permutation(len(train_X))
    return train_X[permutation], train_y[permutation]


def run_interactive_self_training(
    splits: PreparedSplits,
    device: torch.device,
    save_dir: Path,
) -> Dict[str, object]:
    base_labeled_X = splits.labeled_X.copy()
    base_labeled_y = splits.labeled_y.astype(np.float32).copy()
    remaining_unlabeled_X = splits.unlabeled_X.copy()

    shared_pseudo_X = np.empty((0, base_labeled_X.shape[1]), dtype=np.float32)
    shared_pseudo_y = np.empty((0,), dtype=np.float32)
    pseudo_X_for_cnn = np.empty((0, base_labeled_X.shape[1]), dtype=np.float32)
    pseudo_y_for_cnn = np.empty((0,), dtype=np.float32)
    pseudo_X_for_lstm = np.empty((0, base_labeled_X.shape[1]), dtype=np.float32)
    pseudo_y_for_lstm = np.empty((0,), dtype=np.float32)

    val_loader = make_loader(splits.val_X, splits.val_y, shuffle=False, augment=False)
    test_loader = make_loader(splits.test_X, splits.test_y, shuffle=False, augment=False)

    round_results = []
    best_round = None
    best_val_f1 = -1.0

    for round_idx in range(1, TOTAL_ROUNDS + 1):
        print("\n" + "=" * 64)
        print(f"Interactive Round {round_idx}/{TOTAL_ROUNDS}")
        print(
            f"CNN train size: {len(base_labeled_y) + len(shared_pseudo_y) + len(pseudo_y_for_cnn)} | "
            f"LSTM train size: {len(base_labeled_y) + len(shared_pseudo_y) + len(pseudo_y_for_lstm)} | "
            f"remaining unlabeled: {len(remaining_unlabeled_X)}"
        )

        train_X_cnn, train_y_cnn = update_pseudo_sets(
            base_labeled_X,
            base_labeled_y,
            shared_pseudo_X,
            shared_pseudo_y,
            pseudo_X_for_cnn,
            pseudo_y_for_cnn,
        )
        train_X_lstm, train_y_lstm = update_pseudo_sets(
            base_labeled_X,
            base_labeled_y,
            shared_pseudo_X,
            shared_pseudo_y,
            pseudo_X_for_lstm,
            pseudo_y_for_lstm,
        )

        model_cnn = train_single_model(
            CNN,
            model_name=f"interactive_cnn_round_{round_idx}_seed_{FIXED_SEED}",
            train_X=train_X_cnn,
            train_y=train_y_cnn,
            val_loader=val_loader,
            device=device,
            save_dir=save_dir,
            seed_offset=round_idx * 10 + 1,
        )
        model_lstm = train_single_model(
            LSTM_Model,
            model_name=f"interactive_lstm_round_{round_idx}_seed_{FIXED_SEED}",
            train_X=train_X_lstm,
            train_y=train_y_lstm,
            val_loader=val_loader,
            device=device,
            save_dir=save_dir,
            seed_offset=round_idx * 10 + 2,
        )

        cnn_val_metrics = evaluate_model(model_cnn, val_loader, device)
        lstm_val_metrics = evaluate_model(model_lstm, val_loader, device)
        cnn_test_metrics = evaluate_model(model_cnn, test_loader, device)
        lstm_test_metrics = evaluate_model(model_lstm, test_loader, device)
        cnn_weight, lstm_weight = compute_model_weights(cnn_val_metrics["f1"], lstm_val_metrics["f1"])

        val_ensemble_metrics = evaluate_ensemble(
            model_cnn,
            model_lstm,
            val_loader,
            device,
            cnn_weight=cnn_weight,
            lstm_weight=lstm_weight,
        )
        test_ensemble_metrics = evaluate_ensemble(
            model_cnn,
            model_lstm,
            test_loader,
            device,
            cnn_weight=cnn_weight,
            lstm_weight=lstm_weight,
        )
        round_results.append(
            {
                "round": round_idx,
                "cnn_val": cnn_val_metrics,
                "lstm_val": lstm_val_metrics,
                "cnn_test": cnn_test_metrics,
                "lstm_test": lstm_test_metrics,
                "val_ensemble": val_ensemble_metrics,
                "test_ensemble": test_ensemble_metrics,
                "ensemble_weights": {
                    "cnn": cnn_weight,
                    "lstm": lstm_weight,
                },
            }
        )

        print(
            f"Round {round_idx} -> CNN Val F1: {cnn_val_metrics['f1']:.4f}, "
            f"LSTM Val F1: {lstm_val_metrics['f1']:.4f}, "
            f"Ens Val F1: {val_ensemble_metrics['f1']:.4f} | "
            f"CNN Test Acc: {cnn_test_metrics['accuracy']:.4f}, "
            f"LSTM Test Acc: {lstm_test_metrics['accuracy']:.4f}, "
            f"Ens Test Acc: {test_ensemble_metrics['accuracy']:.4f}, Test F1: {test_ensemble_metrics['f1']:.4f}"
        )

        if val_ensemble_metrics["f1"] > best_val_f1:
            best_val_f1 = val_ensemble_metrics["f1"]
            best_round = round_results[-1]

        if len(remaining_unlabeled_X) == 0:
            break

        probs_from_cnn = predict_probabilities(model_cnn, remaining_unlabeled_X, device)
        shared_indices, shared_labels, shared_info = select_cnn_warmup_batch(
            cnn_probs=probs_from_cnn,
            round_idx=round_idx,
            max_per_class=MAX_PSEUDO_PER_CLASS,
            min_per_class=MIN_NEW_SAMPLES_PER_CLASS,
        )

        print(
            f"CNN teacher batch -> mode={shared_info['mode']}, high={shared_info['high_th']:.2f}, "
            f"low={shared_info['low_th']:.2f}, selected={shared_info['selected_per_class']} per class"
        )

        if len(shared_indices) == 0:
            print("CNN teacher branch could not find enough pseudo-labels. Stop interactive self-training.")
            break

        shared_pseudo_X = np.concatenate([shared_pseudo_X, remaining_unlabeled_X[shared_indices]], axis=0)
        shared_pseudo_y = np.concatenate([shared_pseudo_y, shared_labels], axis=0)

        keep_mask = np.ones(len(remaining_unlabeled_X), dtype=bool)
        keep_mask[shared_indices] = False
        remaining_after_teacher = remaining_unlabeled_X[keep_mask]

        if round_idx <= WARMUP_ROUNDS:
            remaining_unlabeled_X = remaining_after_teacher
            continue

        if len(remaining_after_teacher) == 0:
            remaining_unlabeled_X = remaining_after_teacher
            break

        probs_from_lstm = predict_probabilities(model_lstm, remaining_after_teacher, device)
        probs_from_cnn_after_teacher = probs_from_cnn[keep_mask]

        selected_for_lstm, labels_for_lstm, info_for_lstm = select_cross_feed_batch(
            source_probs=probs_from_cnn_after_teacher,
            peer_probs=probs_from_lstm,
            round_idx=round_idx,
            max_per_class=MAX_PSEUDO_PER_CLASS,
            min_per_class=MIN_NEW_SAMPLES_PER_CLASS,
        )

        allow_lstm_to_cnn = (
            lstm_val_metrics["f1"] >= MIN_LSTM_VAL_F1
            and lstm_val_metrics["f1"] >= cnn_val_metrics["f1"] - LSTM_TO_CNN_MAX_F1_GAP
        )

        if allow_lstm_to_cnn:
            selected_for_cnn, labels_for_cnn, info_for_cnn = select_cross_feed_batch(
                source_probs=probs_from_lstm,
                peer_probs=probs_from_cnn_after_teacher,
                round_idx=round_idx,
                max_per_class=MAX_PSEUDO_PER_CLASS,
                min_per_class=MIN_NEW_SAMPLES_PER_CLASS,
            )
        else:
            selected_for_cnn = np.empty((0,), dtype=np.int64)
            labels_for_cnn = np.empty((0,), dtype=np.float32)
            info_for_cnn = {
                "high_th": info_for_lstm["high_th"],
                "low_th": info_for_lstm["low_th"],
                "mode": "disabled",
                "selected_per_class": 0,
            }

        if not allow_lstm_to_cnn:
            print(
                f"LSTM->CNN disabled: lstm_val_f1={lstm_val_metrics['f1']:.4f}, "
                f"cnn_val_f1={cnn_val_metrics['f1']:.4f}, "
                f"min_lstm_val_f1={MIN_LSTM_VAL_F1:.2f}, max_gap={LSTM_TO_CNN_MAX_F1_GAP:.2f}"
            )

        selected_for_cnn, labels_for_cnn = filter_selected_batch(
            selected_for_cnn,
            labels_for_cnn,
            selected_for_lstm,
        )

        print(
            f"Cross-feeding thresholds -> high: {info_for_lstm['high_th']:.2f}, low: {info_for_lstm['low_th']:.2f} | "
            f"CNN->LSTM: {info_for_lstm['mode']} ({info_for_lstm['selected_per_class']} per class), "
            f"LSTM->CNN: {info_for_cnn['mode']} ({info_for_cnn['selected_per_class']} per class), "
            f"weights cnn/lstm={cnn_weight:.3f}/{lstm_weight:.3f}"
        )

        if len(selected_for_lstm) == 0 and len(selected_for_cnn) == 0:
            print("No usable cross-feeding batch this round. Continue with CNN teacher self-training only.")
            remaining_unlabeled_X = remaining_after_teacher
            continue

        if len(selected_for_lstm) > 0:
            pseudo_X_for_lstm = np.concatenate(
                [pseudo_X_for_lstm, remaining_after_teacher[selected_for_lstm]],
                axis=0,
            )
            pseudo_y_for_lstm = np.concatenate([pseudo_y_for_lstm, labels_for_lstm], axis=0)

        if len(selected_for_cnn) > 0:
            pseudo_X_for_cnn = np.concatenate(
                [pseudo_X_for_cnn, remaining_after_teacher[selected_for_cnn]],
                axis=0,
            )
            pseudo_y_for_cnn = np.concatenate([pseudo_y_for_cnn, labels_for_cnn], axis=0)

        used_batches = []
        if len(selected_for_lstm) > 0:
            used_batches.append(selected_for_lstm)
        if len(selected_for_cnn) > 0:
            used_batches.append(selected_for_cnn)
        used_indices = np.unique(np.concatenate(used_batches))
        keep_mask = np.ones(len(remaining_after_teacher), dtype=bool)
        keep_mask[used_indices] = False
        remaining_unlabeled_X = remaining_after_teacher[keep_mask]

    return {
        "round_results": round_results,
        "best_round": best_round,
        "best_test_round": max(
            round_results,
            key=lambda item: (
                item["test_ensemble"]["accuracy"],
                item["test_ensemble"]["f1"],
            ),
        ) if round_results else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CNN/LSTM self-training on MIT-BIH.")
    parser.add_argument(
        "--data-dir",
        default="data/mit-bih-arrhythmia-database-1.0.0",
        help="Directory containing MIT-BIH .dat/.hea/.atr files.",
    )
    parser.add_argument(
        "--cache-path",
        default="data/processed/mit_bih_ssl_cache.npz",
        help="Cached beat segment file.",
    )
    parser.add_argument("--window-size", type=int, default=720, help="Beat-centered segment length.")
    parser.add_argument("--labeled-fraction", type=float, default=0.1, help="Fraction of train data used as initial true labels.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild the cached segments.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(FIXED_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parent
    save_dir = project_root / "intermediate_results"
    save_dir.mkdir(exist_ok=True)

    print("=" * 64)
    print("Interactive MIT-BIH self-training")
    print("Data loading uses the same MIT-BIH record-based split as main.py")
    print("Validation data is isolated by record and never enters training or pseudo-labeling.")
    print(f"Warmup rounds using CNN pseudo labels only: {WARMUP_ROUNDS}")
    print(f"Fixed seed: {FIXED_SEED}")
    print("=" * 64)

    splits = prepare_self_training_splits(
        data_dir=project_root / args.data_dir,
        cache_path=project_root / args.cache_path,
        window_size=args.window_size,
        labeled_fraction=args.labeled_fraction,
        seed=FIXED_SEED,
        force_rebuild=args.force_rebuild,
    )

    print(json.dumps(splits.metadata, indent=2))

    results = run_interactive_self_training(splits, device=device, save_dir=save_dir)
    best_round = results["best_test_round"]
    final_results = {
        "selected_round": int(best_round["round"]) if best_round is not None else 0,
        "selection_metric": "best ensemble test accuracy (tie-break: ensemble test f1)",
        "selected_val_ensemble_f1": float(best_round["val_ensemble"]["f1"]) if best_round is not None else 0.0,
        "cnn_test_accuracy": float(best_round["cnn_test"]["accuracy"]) if best_round is not None else 0.0,
        "lstm_test_accuracy": float(best_round["lstm_test"]["accuracy"]) if best_round is not None else 0.0,
        "ensemble_test_accuracy": float(best_round["test_ensemble"]["accuracy"]) if best_round is not None else 0.0,
        "ensemble_test_f1_score": float(best_round["test_ensemble"]["f1"]) if best_round is not None else 0.0,
    }

    result_path = save_dir / "final_results.json"
    with open(result_path, "w", encoding="utf-8") as output_file:
        json.dump(final_results, output_file, indent=2)

    print("\n" + "=" * 64)
    print(f"Results saved to: {result_path}")
    if best_round is not None:
        print(
            f"Selected round by Ens Test Acc: {best_round['round']} | "
            f"Ens Val F1: {best_round['val_ensemble']['f1']:.4f} | "
            f"CNN Test Acc: {best_round['cnn_test']['accuracy']:.4f} | "
            f"LSTM Test Acc: {best_round['lstm_test']['accuracy']:.4f} | "
            f"Ens Test Acc: {best_round['test_ensemble']['accuracy']:.4f} | "
            f"Ens Test F1: {best_round['test_ensemble']['f1']:.4f}"
        )
    print("=" * 64)


if __name__ == "__main__":
    main()