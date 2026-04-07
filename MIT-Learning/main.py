from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data_processing.mit_bih_ssl import PreparedSplits, prepare_self_training_splits
from model.CNN import CNN


BATCH_SIZE = 64
NUM_EPOCHS = 25
LR = 1e-3
TOTAL_ROUNDS = 5
CONF_TH_HIGH = 0.95
CONF_TH_LOW = 0.05
MAX_PSEUDO_PER_CLASS = 800
MIN_NEW_SAMPLES_PER_CLASS = 32


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


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
				shift = np.random.randint(-24, 25)
				signal = np.roll(signal, shift)

			if np.random.rand() < 0.2:
				mask_len = min(60, len(signal) // 8)
				start = np.random.randint(0, len(signal) - mask_len)
				signal[start:start + mask_len] = 0.0

		signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
		target_tensor = torch.tensor(target).float()
		return signal_tensor, target_tensor


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool, augment: bool = False) -> DataLoader:
	dataset = AugmentedDataset(X, y, augment=augment)
	return DataLoader(
		dataset,
		batch_size=BATCH_SIZE,
		shuffle=shuffle,
		num_workers=0,
		pin_memory=torch.cuda.is_available(),
	)


def train_model(
	model: nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	num_epochs: int,
	device: torch.device,
	save_path: str,
) -> Dict[str, list]:
	history = {
		"train_loss": [],
		"valid_loss": [],
		"train_acc": [],
		"valid_acc": [],
	}
	best_f1 = -1.0

	for epoch in range(num_epochs):
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

		train_loss = float(np.mean(train_losses))
		train_acc = accuracy_score(train_targets, train_preds)

		val_metrics = evaluate_model(model, val_loader, device, criterion)

		history["train_loss"].append(train_loss)
		history["valid_loss"].append(val_metrics["loss"])
		history["train_acc"].append(train_acc)
		history["valid_acc"].append(val_metrics["accuracy"])

		if val_metrics["f1"] > best_f1:
			best_f1 = val_metrics["f1"]
			torch.save(model.state_dict(), save_path)

		if epoch in {10, 15, 20}:
			new_lr = {10: 5e-4, 15: 1e-4, 20: 5e-5}[epoch]
			for group in optimizer.param_groups:
				group["lr"] = new_lr

		print(
			f"Epoch [{epoch + 1}/{num_epochs}] | "
			f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
			f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}"
		)

	return history


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

	preds = (np.asarray(probs) >= 0.5).astype(np.int64)
	targets_array = np.asarray(targets).astype(np.int64)

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
	all_probs = []
	with torch.no_grad():
		for (signals,) in loader:
			signals = signals.to(device)
			outputs = model(signals)
			all_probs.extend(outputs.cpu().numpy().ravel())

	return np.asarray(all_probs, dtype=np.float32)


def select_high_confidence_samples(
	probabilities: np.ndarray,
	max_per_class: int,
	min_per_class: int,
) -> Tuple[np.ndarray, np.ndarray]:
	positive_indices = np.where(probabilities >= CONF_TH_HIGH)[0]
	negative_indices = np.where(probabilities <= CONF_TH_LOW)[0]

	if len(positive_indices) < min_per_class or len(negative_indices) < min_per_class:
		return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

	positive_order = np.argsort(probabilities[positive_indices])[::-1][:max_per_class]
	negative_order = np.argsort(probabilities[negative_indices])[:max_per_class]
	chosen_positive = positive_indices[positive_order]
	chosen_negative = negative_indices[negative_order]

	return np.sort(chosen_positive), np.sort(chosen_negative)


def plot_history(history: Dict[str, list], save_path: str) -> None:
	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(history["train_loss"], label="Train Loss")
	plt.plot(history["valid_loss"], label="Val Loss")
	plt.title("Loss Curve")
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(history["train_acc"], label="Train Acc")
	plt.plot(history["valid_acc"], label="Val Acc")
	plt.title("Accuracy Curve")
	plt.legend()

	plt.tight_layout()
	plt.savefig(save_path)
	plt.close()


def run_self_training(splits: PreparedSplits, device: torch.device, save_dir: Path, seed: int) -> Dict[str, object]:
	criterion = nn.BCELoss()

	base_labeled_X = splits.labeled_X.copy()
	base_labeled_y = splits.labeled_y.copy()
	remaining_unlabeled_X = splits.unlabeled_X.copy()
	remaining_unlabeled_y = splits.unlabeled_y.copy()

	pseudo_X = np.empty((0, base_labeled_X.shape[1]), dtype=np.float32)
	pseudo_y = np.empty((0,), dtype=np.float32)

	val_loader = make_loader(splits.val_X, splits.val_y, shuffle=False, augment=False)
	test_loader = make_loader(splits.test_X, splits.test_y, shuffle=False, augment=False)

	round_results = []
	best_round = None
	best_val_f1 = -1.0
	best_history = None

	for round_idx in range(1, TOTAL_ROUNDS + 1):
		if len(pseudo_X) > 0:
			train_X = np.concatenate([base_labeled_X, pseudo_X], axis=0)
			train_y = np.concatenate([base_labeled_y, pseudo_y], axis=0)
		else:
			train_X = base_labeled_X
			train_y = base_labeled_y

		permutation = np.random.permutation(len(train_X))
		train_X = train_X[permutation]
		train_y = train_y[permutation]

		print("\n" + "=" * 64)
		print(f"Round {round_idx}/{TOTAL_ROUNDS}")
		print(
			f"Train set -> labeled: {len(base_labeled_y)}, pseudo: {len(pseudo_y)}, total: {len(train_y)} | "
			f"remaining unlabeled: {len(remaining_unlabeled_y)}"
		)

		train_loader = make_loader(train_X, train_y, shuffle=True, augment=True)
		model = CNN().to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=LR)

		model_path = save_dir / f"cnn_round_{round_idx}_seed_{seed}.pth"
		history = train_model(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			criterion=criterion,
			optimizer=optimizer,
			num_epochs=NUM_EPOCHS,
			device=device,
			save_path=str(model_path),
		)

		model.load_state_dict(torch.load(model_path, map_location=device))
		val_metrics = evaluate_model(model, val_loader, device, criterion)
		test_metrics = evaluate_model(model, test_loader, device, criterion)
		round_result = {
			"round": round_idx,
			"train_size": int(len(train_y)),
			"pseudo_size": int(len(pseudo_y)),
			"val_metrics": val_metrics,
			"test_metrics": test_metrics,
		}
		round_results.append(round_result)

		print(
			f"Round {round_idx} -> Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f} | "
			f"Test Acc: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['f1']:.4f}"
		)

		if val_metrics["f1"] > best_val_f1:
			best_val_f1 = val_metrics["f1"]
			best_round = round_result
			best_history = history

		if len(remaining_unlabeled_X) == 0:
			break

		probabilities = predict_probabilities(model, remaining_unlabeled_X, device)
		pos_idx, neg_idx = select_high_confidence_samples(
			probabilities,
			max_per_class=MAX_PSEUDO_PER_CLASS,
			min_per_class=MIN_NEW_SAMPLES_PER_CLASS,
		)

		if len(pos_idx) == 0 or len(neg_idx) == 0:
			print("No enough high-confidence samples for the next round. Stop self-training.")
			break

		selected_idx = np.sort(np.concatenate([pos_idx, neg_idx]))
		selected_labels = np.concatenate(
			[
				np.ones(len(pos_idx), dtype=np.float32),
				np.zeros(len(neg_idx), dtype=np.float32),
			]
		)
		selected_X = remaining_unlabeled_X[selected_idx]

		pseudo_X = np.concatenate([pseudo_X, selected_X], axis=0)
		pseudo_y = np.concatenate([pseudo_y, selected_labels], axis=0)

		keep_mask = np.ones(len(remaining_unlabeled_X), dtype=bool)
		keep_mask[selected_idx] = False
		remaining_unlabeled_X = remaining_unlabeled_X[keep_mask]
		remaining_unlabeled_y = remaining_unlabeled_y[keep_mask]

	if best_history is not None:
		plot_history(best_history, str(save_dir / "best_round_training_curve.png"))

	return {
		"metadata": splits.metadata,
		"round_results": round_results,
		"best_round": best_round,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="MIT-BIH CNN self-training with a small labeled subset.")
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
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument("--force-rebuild", action="store_true", help="Rebuild the cached segments.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	project_root = Path(__file__).resolve().parent
	save_dir = project_root / "results"
	save_dir.mkdir(exist_ok=True)

	print("=" * 64)
	print("Preparing MIT-BIH self-training dataset")
	print("Task: normal beat vs abnormal beat")
	print("Step 1 is replaced with a small true-labeled subset from the training records.")
	print("=" * 64)

	splits = prepare_self_training_splits(
		data_dir=project_root / args.data_dir,
		cache_path=project_root / args.cache_path,
		window_size=args.window_size,
		labeled_fraction=args.labeled_fraction,
		seed=args.seed,
		force_rebuild=args.force_rebuild,
	)

	print(json.dumps(splits.metadata, indent=2))

	results = run_self_training(splits, device=device, save_dir=save_dir, seed=args.seed)
	result_path = save_dir / "mit_bih_ssl_results.json"
	with open(result_path, "w", encoding="utf-8") as output_file:
		json.dump(results, output_file, indent=2)

	best_round = results["best_round"]
	print("\n" + "=" * 64)
	print(f"Results saved to: {result_path}")
	if best_round is not None:
		print(
			f"Best round: {best_round['round']} | "
			f"Val F1: {best_round['val_metrics']['f1']:.4f} | "
			f"Test F1: {best_round['test_metrics']['f1']:.4f}"
		)
	print("=" * 64)


if __name__ == "__main__":
	main()
