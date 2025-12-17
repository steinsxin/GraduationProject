import os
import numpy as np
from data_processing.Dealdata import ECG_Datadeal
from scipy.signal import filtfilt, butter, find_peaks
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import my_func
from model.CNN import CNN


# ============================================================
# 全局参数
# ============================================================
FS = 400
CV_TH = 0.22
ARI_TH = 0.24
CONF_THRESHOLD = 60.0

BATCH_SIZE = 32
NUM_EPOCHS = 40
LR = 1e-3


# ============================================================
# 信号处理 & 特征
# ============================================================
def _bandpass(sig, fs=FS, low=5, high=45):
    b, a = butter(3, [low, high], btype='band', fs=fs)
    return filtfilt(b, a, sig, padlen=0)


def _detect_qrs(sig, fs=FS):
    win = int(0.2 * fs)
    diff = np.r_[0, 0, sig[2:] - sig[:-2]]

    pos = diff[diff > 0]
    if len(pos) == 0:
        return np.array([])

    th = max(np.percentile(pos, 75) * 1.2,
             np.percentile(np.abs(sig), 75) * 0.4)

    peaks, _ = find_peaks(diff, height=th, distance=win)
    return peaks


def _calculate_features(ecg):
    sig = _bandpass(ecg)
    r = _detect_qrs(sig)
    rr = np.diff(r) / FS

    if len(rr) < 5:
        return 0.0, 0.0

    mean_rr = np.mean(rr)
    if mean_rr == 0:
        return 0.0, 0.0

    cv = np.std(rr) / mean_rr
    ari = np.sum(np.abs(np.diff(rr))) / np.sum(rr)
    return cv, ari


def calculate_confidence(cv, ari, cv_th, ari_th):
    is_afib = (cv > cv_th) and (ari > ari_th)

    if is_afib:
        score = min((cv - cv_th) / cv_th,
                    (ari - ari_th) / ari_th)
        conf = 50 + 50 * score
        label = 1
    else:
        score = max((cv_th - cv) / cv_th,
                    (ari_th - ari) / ari_th)
        conf = 50 + 50 * score
        label = 0

    conf = np.clip(conf, 50, 100)
    return label, conf


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":

    print("\n================ Part 0: 数据加载 =================")

    train_npy_path = ECG_Datadeal(os.path.join("data", "train", "traindata.mat"))
    all_data = np.load(train_npy_path)

    label_data = all_data[:1000]
    no_label_data = all_data[1000:]

    labels = np.array([1] * 500 + [0] * 500)

    print(f"有标签数据: {label_data.shape}")
    print(f"无标签数据: {no_label_data.shape}")

    # ========================================================
    # Part 1：无标签数据 → 伪标签
    # ========================================================
    print("\n================ Part 1: 伪标签生成 =================")

    features = np.array(Parallel(n_jobs=-1)(
        delayed(_calculate_features)(no_label_data[i])
        for i in tqdm(range(len(no_label_data)), desc="特征提取")
    ))

    results = [calculate_confidence(f[0], f[1], CV_TH, ARI_TH) for f in features]

    afib_idx = [i for i, (p, c) in enumerate(results) if p == 1 and c >= CONF_THRESHOLD]
    normal_idx = [i for i, (p, c) in enumerate(results) if p == 0 and c >= CONF_THRESHOLD]

    X_afib = no_label_data[afib_idx]
    X_normal = no_label_data[normal_idx]

    print(f"伪 AFib: {len(X_afib)}")
    print(f"伪 Normal: {len(X_normal)}")

    if len(X_afib) == 0 or len(X_normal) == 0:
        raise RuntimeError("⚠️ 伪标签严重失衡，请降低 CONF_THRESHOLD")

    X_train = np.concatenate([X_afib, X_normal])
    y_train = np.concatenate([
        np.ones(len(X_afib)),
        np.zeros(len(X_normal))
    ])

    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    # ========================================================
    # Part 2：Labeled Seed / Test（真实标签）
    # ========================================================
    print("\n================ Part 2: Labeled Seed / Test =================")

    X_labeled, X_test, y_labeled, y_test = train_test_split(
        label_data, labels,
        test_size=0.8,
        stratify=labels,
        random_state=42
    )

    print(f"Labeled (for training): {X_labeled.shape}, Test: {X_test.shape}")

    # ========================================================
    # Part 3：PyTorch 训练
    # ========================================================
    print("\n================ Part 3: CNN Training =================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_loader(X, y, shuffle):
        X = torch.from_numpy(X).float().unsqueeze(1)
        y = torch.from_numpy(y).float()
        return DataLoader(
            TensorDataset(X, y),
            batch_size=BATCH_SIZE,
            shuffle=shuffle
        )

    # 混合伪标签数据和有标签的种子数据
    print(f"混合伪标签 ({X_train.shape[0]}) 和真实标签 ({X_labeled.shape[0]}) 数据...")
    X_combined = np.concatenate([X_train, X_labeled])
    y_combined = np.concatenate([y_train, y_labeled])
    print(f"总数据: {X_combined.shape}")

    # 将混合后的数据划分为训练集和验证集 (4:1)
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_combined, y_combined,
        test_size=0.2,  # 20% for validation (4:1 split)
        stratify=y_combined,
        random_state=42
    )
    print(f"最终训练集: {X_train_final.shape}, 验证集: {X_val.shape}")

    train_loader = make_loader(X_train_final, y_train_final, shuffle=True)
    val_loader = make_loader(X_val, y_val, False)

    model = CNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    save_path = "save/Bestmodel_SemiSupervised_CNN.pth"
    os.makedirs("save", exist_ok=True)

    history = my_func.train_model(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,  # 使用新划分的验证集
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_best=True,
        save_path=save_path
    )

    # ========================================================
    # Part 4: 保存训练过程图
    # ========================================================
    print("\n================ Part 4: Saving Training Process Plot =================")

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['valid_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['valid_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    save_fig_path = os.path.join("results", "training_process.png")
    plt.savefig(save_fig_path)
    print(f"Training process plot saved to '{save_fig_path}'")


    # ========================================================
    # Part 5：最终测试
    # ========================================================
    print("\n================ Part 5: Final Test =================")

    state_dict = torch.load(save_path, weights_only=True)
    model.load_state_dict(state_dict)

    test_loader = make_loader(X_test, y_test, False)

    preds, gts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x).squeeze()
            pred = (out > 0.5).int().cpu().numpy()
            preds.extend(pred)
            gts.extend(y.numpy())

    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds)

    print("\n====== Final Result ======")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
