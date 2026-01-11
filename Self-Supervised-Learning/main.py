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
    # 定义通用训练函数
    # ========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_loader(X, y, shuffle):
        X = torch.from_numpy(X).float().unsqueeze(1)
        y = torch.from_numpy(y).float()
        return DataLoader(
            TensorDataset(X, y),
            batch_size=BATCH_SIZE,
            shuffle=shuffle
        )

    def run_cnn_training(X_train, y_train, model_name="Bestmodel_CNN"):
        print(f"\n>>> Start Training: {model_name}")
        
        # 将伪标签数据划分为 训练集(80%) 和 验证集(20%)
        # 用于 Early Stopping，不使用真实标签数据进行验证
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=42
        )
        print(f"   Train: {X_tr.shape}, Val: {X_val.shape}")

        train_loader = make_loader(X_tr, y_tr, shuffle=True)
        val_loader = make_loader(X_val, y_val, False)
        
        model = CNN().to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        save_dir = "save"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name}.pth")
        
        history = my_func.train_model(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=NUM_EPOCHS,
            device=device,
            save_best=True,
            save_path=save_path
        )
        return model, save_path, history

    # ========================================================
    # Part 3：Iterative Training (Round 1: Rule-based)
    # ========================================================
    print("\n================ Part 3: Iterative Round 1 (Rule-based) =================")
    
    model_r1, path_r1, hist_r1 = run_cnn_training(X_train, y_train, "CNN_Step1_RuleBased")
    
    # Eval Round 1
    model_r1.load_state_dict(torch.load(path_r1, map_location=device))
    test_loader = make_loader(X_test, y_test, False)
    
    preds, gts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model_r1(x).squeeze()
            preds.extend((out > 0.5).int().cpu().numpy())
            gts.extend(y.numpy())
            
    print(f"Round 1 Result (Rule-based) -> Acc: {accuracy_score(gts, preds):.4f}, F1: {f1_score(gts, preds):.4f}")

    # ========================================================
    # Part 4：Iterative Round 2 (Model-based Refinement)
    # ========================================================
    print("\n================ Part 4: Iterative Round 2 (Model Refinement) =================")
    print("Using Round 1 model to predict on ALL unlabeled data...")
    
    # 1. 对所有无标签数据进行预测
    X_all_unlabeled = torch.from_numpy(no_label_data).float().unsqueeze(1)
    unlabeled_loader = DataLoader(TensorDataset(X_all_unlabeled), batch_size=BATCH_SIZE, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for (x,) in tqdm(unlabeled_loader, desc="Inference"):
            x = x.to(device)
            out = model_r1(x).view(-1)
            all_probs.extend(out.cpu().numpy())
            
    all_probs = np.array(all_probs)
    
    # 2. 选取 Model 非常确定的样本 (Confidence > 0.95 or < 0.05)
    # 这能过滤掉 Rule 认为是对的但 Model 觉得长得像噪声的样本
    CONF_TH_HIGH = 0.95
    CONF_TH_LOW = 0.05
    
    afib_model_idx = np.where(all_probs > CONF_TH_HIGH)[0]
    normal_model_idx = np.where(all_probs < CONF_TH_LOW)[0]
    
    print(f"Model High Conf Candidates -> AFib: {len(afib_model_idx)}, Normal: {len(normal_model_idx)}")
    
    # 3. 平衡 & Top-K 选取
    target_count_r2 = min(len(afib_model_idx), len(normal_model_idx), 2500) # 稍微放宽一点上限
    print(f"Target count per class (Round 2): {target_count_r2}")
    
    # 对 AFib (prob 接近 1.0)
    afib_probs_subset = all_probs[afib_model_idx]
    # 降序取前K
    afib_top_k_indices = np.argsort(afib_probs_subset)[::-1][:target_count_r2] 
    final_afib_idx_r2 = afib_model_idx[afib_top_k_indices]
    
    # 对 Normal (prob 接近 0.0)
    normal_probs_subset = all_probs[normal_model_idx]
    # 升序取前K (越小越好)
    normal_top_k_indices = np.argsort(normal_probs_subset)[:target_count_r2] 
    final_normal_idx_r2 = normal_model_idx[normal_top_k_indices]
    
    X_afib_r2 = no_label_data[final_afib_idx_r2]
    X_normal_r2 = no_label_data[final_normal_idx_r2]
    
    X_train_r2 = np.concatenate([X_afib_r2, X_normal_r2])
    y_train_r2 = np.concatenate([np.ones(len(X_afib_r2)), np.zeros(len(X_normal_r2))])
    
    perm = np.random.permutation(len(X_train_r2))
    X_train_r2, y_train_r2 = X_train_r2[perm], y_train_r2[perm]
    
    print(f"Final Training Set for Round 2: {X_train_r2.shape}")

    # 4. Train Final Model
    model_final, path_final, hist_final = run_cnn_training(X_train_r2, y_train_r2, "CNN_Final_iter2")

    # ========================================================
    # Part 5：最终测试
    # ========================================================
    print("\n================ Part 5: Final Test (Round 2 Model) =================")

    model_final.load_state_dict(torch.load(path_final, map_location=device))

    preds, gts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model_final(x).squeeze()
            pred = (out > 0.5).int().cpu().numpy()
            preds.extend(pred)
            gts.extend(y.numpy())

    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds)

    print("\n====== Final Result (Refined) ======")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    
    # Save Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist_final['train_loss'], label='Train Loss')
    plt.plot(hist_final['valid_loss'], label='Validation Loss')
    plt.title('Loss Curve (Final)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist_final['train_acc'], label='Train Accuracy')
    plt.plot(hist_final['valid_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curve (Final)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", "training_process_final.png"))
    print("Plot saved.")
