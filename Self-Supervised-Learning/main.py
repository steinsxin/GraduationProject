import os
import json
import numpy as np
from data_processing.Dealdata import ECG_Datadeal
from scipy.signal import filtfilt, butter, find_peaks
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
NUM_EPOCHS = 60
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
# 数据增强 Dataset
# ============================================================
class AugmentedDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 复制数据防止修改原数组
        sig = self.X[idx].copy()
        target = self.y[idx]

        if self.augment:
            # 1. 随机幅度缩放 (Scaling)
            # 模拟信号强度的变化
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.8, 1.2)
                sig = sig * scale
            
            # 2. 随机高斯噪声 (Jittering)
            # 模拟传感器噪声
            if np.random.rand() < 0.5:
                # 假设数据已经过标准化 (mean=0, std=1)，添加 0.05 std 的噪声
                noise = np.random.normal(0, 0.05, sig.shape)
                sig = sig + noise

            # 3. 随机平移 (Time Shift/Roll)
            # 模拟截取窗口的偏移
            if np.random.rand() < 0.5:
                # 最大移动 ±100 个采样点 (FS=400, 0.25s)
                shift = np.random.randint(-100, 100)
                sig = np.roll(sig, shift)
                # 注意：np.roll 是循环的，但对于心电这种周期或者长信号通常影响不大
                # 如果是边界敏感的，可以使用 pad + crop 方式，简单起见这里用 roll

            # 4. Random Masking (Cutout)
            # 随机遮挡 0.5s 的信号 (200采样点)
            if np.random.rand() < 0.3:
                mask_len = 200
                start_p = np.random.randint(0, len(sig) - mask_len)
                sig[start_p:start_p+mask_len] = 0.0

        # 转为 Tensor: (1, Length)
        sig_tensor = torch.from_numpy(sig).float().unsqueeze(0)
        target_tensor = torch.tensor(target).float()
        
        return sig_tensor, target_tensor


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

    def make_loader(X, y, shuffle, augment=False):
        dataset = AugmentedDataset(X, y, augment=augment)
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle
        )

    def run_cnn_training(X_train, y_train, model_name="Bestmodel_CNN", seed=42):
        print(f"\n>>> Start Training: {model_name} (Seed: {seed})")
        
        # 将伪标签数据划分为 训练集(80%) 和 验证集(20%)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=seed
        )
        print(f"   Train: {X_tr.shape}, Val: {X_val.shape}")

        train_loader = make_loader(X_tr, y_tr, shuffle=True, augment=True)
        val_loader = make_loader(X_val, y_val, False, augment=False)
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        model = CNN().to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        save_dir = "save"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name}_seed{seed}.pth")
        
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
    # 多次运行的主循环
    # ========================================================
    NUM_RUNS = 1
    
    # Metrics Storage
    all_rule_accs = []      # Pure Rule-based (CV/ARI) Accuracy on Test Set
    all_r1_accs = []        # Round 1 Model Accuracy
    all_final_accs = []     # Round 2 (Final) Model Accuracy
    
    all_r1_f1s = []
    all_final_f1s = []

    # Pre-calculate Rule-based Performance on Test Set (One-time)
    print("\n--- Evaluating Pure Rule-based Performance on Test Set ---")
    
    # 提取测试集特征（如果有标签数据量大，这里可能会花点时间）
    test_features = np.array(Parallel(n_jobs=-1)(
        delayed(_calculate_features)(X_test[i])
        for i in tqdm(range(len(X_test)), desc="Test Feat Extraction")
    ))
    
    # 应用规则
    rule_test_results = np.array([calculate_confidence(f[0], f[1], CV_TH, ARI_TH) for f in test_features])
    rule_preds = rule_test_results[:, 0] # 0 or 1
    
    rule_acc = accuracy_score(y_test, rule_preds)
    rule_f1 = f1_score(y_test, rule_preds)
    print(f"Pure Rule-based Result -> Acc: {rule_acc:.4f}, F1: {rule_f1:.4f}")

    for run_idx in range(NUM_RUNS):
        current_seed = 42 + run_idx
        print(f"\n\n{'#'*60}")
        print(f"### RUN {run_idx + 1}/{NUM_RUNS} (Seed: {current_seed})")
        print(f"{'#'*60}")
        
        # Store Rule Acc (Repeated for consistency across runs structure)
        all_rule_accs.append(rule_acc)

        # ========================================================
        # Part 3：Iterative Self-Training (15 Rounds)
        # ========================================================
        # 定义总共的轮次
        TOTAL_ROUNDS = 15
        
        # 变量初始化
        current_model = None
        current_model_path = ""
        last_hist = None
        
        # 为了记录过程，我们存一下每一轮的 ACC
        run_iter_accs = []
        run_iter_f1s = []
        
        for r_idx in range(1, TOTAL_ROUNDS + 1):
            print(f"\n--- Iterative Round {r_idx} / {TOTAL_ROUNDS} ---")
            
            # --- 数据准备 ---
            if r_idx == 1:
                # Round 1: 使用 Rule-based 生成的伪标签 (Part 1 结果)
                curr_X_train = X_train
                curr_y_train = y_train
                model_suffix = "Iter1_RuleBased"
                
            else:
                # Round 2+: 使用上一轮模型生成伪标签 (Self-Training)
                print(f"Using Round {r_idx-1} model to generate pseudo-labels...")
                
                # 加载上一轮模型
                current_model.load_state_dict(torch.load(current_model_path, map_location=device, weights_only=True))
                current_model.eval()
                
                # 1. 预测无标签数据
                X_all_unlabeled = torch.from_numpy(no_label_data).float().unsqueeze(1)
                unlabeled_loader = DataLoader(TensorDataset(X_all_unlabeled), batch_size=BATCH_SIZE, shuffle=False)
                
                all_probs = []
                with torch.no_grad():
                    for (x,) in tqdm(unlabeled_loader, desc=f"Inference R{r_idx-1}", leave=False):
                        x = x.to(device)
                        out = current_model(x).view(-1)
                        all_probs.extend(out.cpu().numpy())
                all_probs = np.array(all_probs)
                
                # 2. 筛选高置信度样本
                CONF_TH_HIGH = 0.95
                CONF_TH_LOW = 0.05
                
                afib_model_idx = np.where(all_probs > CONF_TH_HIGH)[0]
                normal_model_idx = np.where(all_probs < CONF_TH_LOW)[0]
                
                # 3. 平衡 & Top-K 选取
                target_count = min(len(afib_model_idx), len(normal_model_idx), 2500)
                
                if target_count < 50:
                    print(f"⚠️ Warning: Not enough high-confidence samples in Round {r_idx}. Skipping...")
                    # 如果样本太少，就中止后续轮次
                    break

                # AFib (prob -> 1.0)
                afib_probs_subset = all_probs[afib_model_idx]
                afib_top_k = np.argsort(afib_probs_subset)[::-1][:target_count] 
                final_afib_idx = afib_model_idx[afib_top_k]
                
                # Normal (prob -> 0.0)
                normal_probs_subset = all_probs[normal_model_idx]
                normal_top_k = np.argsort(normal_probs_subset)[:target_count] 
                final_normal_idx = normal_model_idx[normal_top_k]
                
                X_afib_new = no_label_data[final_afib_idx]
                X_normal_new = no_label_data[final_normal_idx]
                
                print(f"   Round {r_idx} Training Data: {len(X_afib_new)} AFib + {len(X_normal_new)} Normal")
                
                # 4. 构建新一轮训练集
                curr_X_train = np.concatenate([X_afib_new, X_normal_new])
                curr_y_train = np.concatenate([np.ones(len(X_afib_new)), np.zeros(len(X_normal_new))])
                
                perm = np.random.permutation(len(curr_X_train))
                curr_X_train, curr_y_train = curr_X_train[perm], curr_y_train[perm]
                
                model_suffix = f"Iter{r_idx}_SelfTrain"

            # --- 模型训练 ---
            save_name = f"CNN_{model_suffix}"
            current_model, current_model_path, hist = run_cnn_training(
                curr_X_train, curr_y_train, save_name, seed=current_seed
            )
            last_hist = hist
            
            # --- 测试集评估 ---
            current_model.load_state_dict(torch.load(current_model_path, map_location=device, weights_only=True))
            test_loader = make_loader(X_test, y_test, False, augment=False)
            
            preds, gts = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    out = current_model(x).squeeze()
                    preds.extend((out > 0.5).int().cpu().numpy())
                    gts.extend(y.numpy())
            
            acc = accuracy_score(gts, preds)
            f1 = f1_score(gts, preds)
            
            print(f"   [Round {r_idx} Result] -> Acc: {acc:.4f}, F1: {f1:.4f}")
            run_iter_accs.append(acc)
            run_iter_f1s.append(f1)
            
            if r_idx == 1:
                all_r1_accs.append(acc)
                all_r1_f1s.append(f1)
        
        # 循环结束，记录最终结果 (最后一轮)
        if len(run_iter_accs) > 0:
            final_acc = run_iter_accs[-1]
            final_f1 = run_iter_f1s[-1]
        else:
            final_acc = 0
            final_f1 = 0
        
        all_final_accs.append(final_acc)
        all_final_f1s.append(final_f1)
        
        print(f"   Run {run_idx+1} Final Result -> Accuracy: {final_acc:.4f}, F1: {final_f1:.4f}")
        
        # Save Plot for last run only
        if run_idx == 0 and last_hist is not None:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(last_hist['train_loss'], label='Train Loss')
            plt.plot(last_hist['valid_loss'], label='Validation Loss')
            plt.title(f'Loss Curve (Run {run_idx+1}, Final)')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(last_hist['train_acc'], label='Train Accuracy')
            plt.plot(last_hist['valid_acc'], label='Validation Accuracy')
            plt.title(f'Accuracy Curve (Run {run_idx+1}, Final)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("results", f"training_process_run{run_idx+1}.png"))
            plt.close()

    # ========================================================
    # Summary
    # ========================================================
    print("\n\n========================================================")
    print(f"Final Summary Over {NUM_RUNS} Runs")
    print("========================================================")
    
    # Calculate Means & Stds
    mean_rule_acc, std_rule_acc = np.mean(all_rule_accs), np.std(all_rule_accs)
    mean_r1_acc, std_r1_acc = np.mean(all_r1_accs), np.std(all_r1_accs)
    mean_final_acc, std_final_acc = np.mean(all_final_accs), np.std(all_final_accs)
    
    mean_r1_f1, std_r1_f1 = np.mean(all_r1_f1s), np.std(all_r1_f1s)
    mean_final_f1, std_final_f1 = np.mean(all_final_f1s), np.std(all_final_f1s)

    print(f"1. Pure Rule-based Acc: {mean_rule_acc:.4f} (fixed)")
    print(f"2. Round 1 Model Acc:   {mean_r1_acc:.4f} (+/- {std_r1_acc:.4f})")
    print(f"3. Final Model Acc:     {mean_final_acc:.4f} (+/- {std_final_acc:.4f})")

    # Save results to JSON file
    results_data = {
        "rule_based": {
            "mean_accuracy": float(mean_rule_acc),
            "mean_f1_score": float(rule_f1) # rule based is deterministic
        },
        "round_1_model": {
            "mean_accuracy": float(mean_r1_acc),
            "std_accuracy": float(std_r1_acc),
            "mean_f1_score": float(mean_r1_f1),
            "std_f1_score": float(std_r1_f1),
            "raw_accuracies": [float(x) for x in all_r1_accs]
        },
        "final_model": {
            "mean_accuracy": float(mean_final_acc),
            "std_accuracy": float(std_final_acc),
            "mean_f1_score": float(mean_final_f1),
            "std_f1_score": float(std_final_f1),
            "raw_accuracies": [float(x) for x in all_final_accs],
            "raw_f1_scores": [float(x) for x in all_final_f1s]
        }
    }
    
    json_path = os.path.join("results", "final_stats.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=4)
        
    print(f"Detailed results saved to {json_path}")

