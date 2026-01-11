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
CV_TH = 0.02
ARI_TH = 0.24
CONF_THRESHOLD = 50.0

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
    # Part 1：无标签数据 → 伪标签 (Rule-based High Quality Selection)
    # ========================================================
    print("\n================ Part 1: 伪标签生成 (Unsupervised Rule-based) =================")
    
    # 提取无标签数据特征
    print("Step 1: 提取无标签数据特征...")
    unlabeled_features = np.array(Parallel(n_jobs=-1)(
        delayed(_calculate_features)(no_label_data[i])
        for i in tqdm(range(len(no_label_data)), desc="Unlabeled Feat")
    ))

    # 基于专家规则 (CV, ARI) 计算置信度
    # calculated_confidence 返回: (预测类别 0/1, 置信度 50-100)
    print("Step 2: 计算专家规则置信度 (CV & ARI)...")
    rule_results = np.array([calculate_confidence(f[0], f[1], CV_TH, ARI_TH) for f in unlabeled_features])
    
    pred_labels = rule_results[:, 0]
    confidences = rule_results[:, 1]
    
    # 筛选潜在样本
    # 提高基础阈值，例如要求置信度 > 80 (原代码是 50)
    # AFib 样本：类别 1，置信度高
    # Normal 样本：类别 0，置信度高
    
    afib_candidates = np.where(pred_labels == 1)[0]
    normal_candidates = np.where(pred_labels == 0)[0]
    
    print(f"Candidates (Pre-filter) -> AFib: {len(afib_candidates)}, Normal: {len(normal_candidates)}")
    
    # Step 3: Top-K 高质量筛选与平衡 (Quality Control)
    # 策略：取两类中数量较少者的 Top N，或者设定固定上限，取置信度最高的样本
    
    # 设定一个严格的“入围”门槛，比如 85 分以上
    STRICT_TH = 85.0
    afib_qualified = afib_candidates[confidences[afib_candidates] >= STRICT_TH]
    normal_qualified = normal_candidates[confidences[normal_candidates] >= STRICT_TH]
    
    print(f"Qualified (> {STRICT_TH} conf) -> AFib: {len(afib_qualified)}, Normal: {len(normal_qualified)}")
    
    # 确定最终数量：取两者最小值，进行严格平衡
    final_count = min(len(afib_qualified), len(normal_qualified))
    # 可选：再加一个上限，防止伪标签太多
    MAX_COUNT = 2000
    final_count = min(final_count, MAX_COUNT)
    
    print(f"Target count per class: {final_count}")
    
    if final_count == 0:
         raise RuntimeError("⚠️ 没有足够的高置信度样本，请检查数据或降低 STRICT_TH。")

    # 对 AFib 样本按置信度从高到低排序，取前 final_count 个
    afib_sorted_idx = np.argsort(confidences[afib_qualified])[::-1] # 降序
    final_afib_idx = afib_qualified[afib_sorted_idx[:final_count]]
    
    # 对 Normal 样本同理
    normal_sorted_idx = np.argsort(confidences[normal_qualified])[::-1]
    final_normal_idx = normal_qualified[normal_sorted_idx[:final_count]]
    
    X_afib = no_label_data[final_afib_idx]
    X_normal = no_label_data[final_normal_idx]

    print(f"最终选定高质量伪标签: AFib: {len(X_afib)} (Min Conf: {confidences[final_afib_idx[-1]]:.2f}), "
          f"Normal: {len(X_normal)} (Min Conf: {confidences[final_normal_idx[-1]]:.2f})")

    X_train = np.concatenate([X_afib, X_normal])
    y_train = np.concatenate([
        np.ones(len(X_afib)),
        np.zeros(len(X_normal))
    ])

    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    # ========================================================
    # Part 2：Test Set Definition
    # ========================================================
    print("\n================ Part 2: Defining Test Set =================")
    
    # 所有有标签数据作为最终测试集
    X_test = label_data
    y_test = labels
    print(f"Test Set (All Labeled Data): {X_test.shape}")

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

    print(f"仅使用伪标签数据进行训练和验证划分...")
    
    # 将伪标签数据划分为 训练集(80%) 和 验证集(20%)
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    print(f"训练集 (Pseudo Train): {X_train_final.shape}")
    print(f"验证集 (Pseudo Val):   {X_val.shape}")

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
