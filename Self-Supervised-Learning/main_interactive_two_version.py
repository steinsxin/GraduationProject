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
from model.LSTM import LSTM_Model  # New Architecture

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

INTERMEDIATE_DIR = "intermediate_results"
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

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
        sig = self.X[idx].copy()
        target = self.y[idx]

        if self.augment:
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.8, 1.2)
                sig = sig * scale
            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.05, sig.shape)
                sig = sig + noise
            if np.random.rand() < 0.5:
                shift = np.random.randint(-100, 100)
                sig = np.roll(sig, shift)
            if np.random.rand() < 0.3:
                mask_len = 200
                start_p = np.random.randint(0, len(sig) - mask_len)
                sig[start_p:start_p+mask_len] = 0.0

        # CNN expects (1, Length)
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
    # Part 1: Rule-based Initial Selection
    # ========================================================
    print("\n================ Part 1: 伪标签生成 (Unsupervised Rule-based) =================")
    
    unlabeled_features = np.array(Parallel(n_jobs=-1)(
        delayed(_calculate_features)(no_label_data[i])
        for i in tqdm(range(len(no_label_data)), desc="Unlabeled Feat")
    ))

    rule_results = np.array([calculate_confidence(f[0], f[1], CV_TH, ARI_TH) for f in unlabeled_features])
    
    pred_labels = rule_results[:, 0]
    confidences = rule_results[:, 1]
    
    afib_candidates = np.where(pred_labels == 1)[0]
    normal_candidates = np.where(pred_labels == 0)[0]
    
    STRICT_TH = 85.0
    afib_qualified = afib_candidates[confidences[afib_candidates] >= STRICT_TH]
    normal_qualified = normal_candidates[confidences[normal_candidates] >= STRICT_TH]
    
    final_count = min(len(afib_qualified), len(normal_qualified))
    MAX_COUNT = 2000
    final_count = min(final_count, MAX_COUNT)
    
    if final_count == 0:
         raise RuntimeError("⚠️ 没有足够的高置信度样本，请检查数据或降低 STRICT_TH。")

    afib_sorted_idx = np.argsort(confidences[afib_qualified])[::-1]
    final_afib_idx = afib_qualified[afib_sorted_idx[:final_count]]
    
    normal_sorted_idx = np.argsort(confidences[normal_qualified])[::-1]
    final_normal_idx = normal_qualified[normal_sorted_idx[:final_count]]
    
    X_afib = no_label_data[final_afib_idx]
    X_normal = no_label_data[final_normal_idx]

    print(f"Initial Pseudo Labels: AFib: {len(X_afib)}, Normal: {len(X_normal)}")
    X_train_init = np.concatenate([X_afib, X_normal])
    y_train_init = np.concatenate([np.ones(len(X_afib)), np.zeros(len(X_normal))])

    # Shuffle
    perm = np.random.permutation(len(X_train_init))
    X_train_init, y_train_init = X_train_init[perm], y_train_init[perm]

    # ========================================================
    # Part 2: Test Set
    # ========================================================
    X_test = label_data
    y_test = labels
    print(f"Test Set: {X_test.shape}")

    # ========================================================
    # Training Functions
    # ========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    def make_loader(X, y, shuffle, augment=False):
        dataset = AugmentedDataset(X, y, augment=augment)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    def run_training(model_class, X_train, y_train, model_name, seed=42):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 设置numpy的随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        print(f"\n   >>> Training {model_name} (Seed: {seed}) ...")
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
        )

        train_loader = make_loader(X_tr, y_tr, shuffle=True, augment=True)
        val_loader = make_loader(X_val, y_val, False, augment=False)
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        model = model_class().to(device)
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
        # Load best model for return
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        return model, save_path

    # ========================================================
    # Interactive Iteration (Part 3)
    # ========================================================
    NUM_RUNS = 1
    TOTAL_ROUNDS = 10
    
    # Results storage
    results_history = [] 

    for run_idx in range(NUM_RUNS):
        current_seed = 42 + run_idx
        np.random.seed(current_seed)
        print(f"\n\n{'#'*60}")
        print(f"### RUN {run_idx + 1}/{NUM_RUNS} (Seed: {current_seed})")
        print(f"{'#'*60}")
        
        # In iteration 1, both models start with Rule-based Labels
        curr_X_train_cnn = X_train_init.copy()
        curr_y_train_cnn = y_train_init.copy()
        
        curr_X_train_lstm = X_train_init.copy()
        curr_y_train_lstm = y_train_init.copy()
        
        run_res = {
            "cnn_acc": [], "lstm_acc": [], "ensemble_acc": []
        }

        for r_idx in range(1, TOTAL_ROUNDS + 1):
            print(f"\n--- Round {r_idx} / {TOTAL_ROUNDS} (Interactive) ---")
            
            # 1. Train CNN
            model_cnn, path_cnn = run_training(CNN, curr_X_train_cnn, curr_y_train_cnn, f"Inter_CNN_R{r_idx}", current_seed)
            
            # 2. Train LSTM
            model_lstm, path_lstm = run_training(LSTM_Model, curr_X_train_lstm, curr_y_train_lstm, f"Inter_LSTM_R{r_idx}", current_seed)
            
            # 3. Evaluate on Test (Ensemble)
            test_loader = make_loader(X_test, y_test, False, augment=False)
            model_cnn.eval()
            model_lstm.eval()
            
            probs_cnn_test = []
            probs_lstm_test = []
            gts = []
            
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    # CNN Output
                    out_c = model_cnn(x).squeeze()
                    probs_cnn_test.extend(out_c.cpu().numpy())
                    # LSTM Output
                    out_l = model_lstm(x).squeeze()
                    probs_lstm_test.extend(out_l.cpu().numpy())
                    
                    gts.extend(y.numpy())
            
            probs_cnn_test = np.array(probs_cnn_test)
            probs_lstm_test = np.array(probs_lstm_test)
            gts = np.array(gts)
            
            acc_cnn = accuracy_score(gts, (probs_cnn_test > 0.5).astype(int))
            acc_lstm = accuracy_score(gts, (probs_lstm_test > 0.5).astype(int))
            
            # Ensemble (Average Probability)
            probs_ens = (probs_cnn_test + probs_lstm_test) / 2.0
            acc_ens = accuracy_score(gts, (probs_ens > 0.5).astype(int))
            f1_ens = f1_score(gts, (probs_ens > 0.5).astype(int))
            
            print(f"   [Round {r_idx} Result]")
            print(f"     CNN Acc: {acc_cnn:.4f}")
            print(f"     LSTM Acc: {acc_lstm:.4f}")
            print(f"     Ensemble Acc: {acc_ens:.4f}  (F1: {f1_ens:.4f})")
            
            run_res["cnn_acc"].append(acc_cnn)
            run_res["lstm_acc"].append(acc_lstm)
            run_res["ensemble_acc"].append(acc_ens)
            
            # 4. Interactive Pseudo-Labeling (Cross-Feeding)
            # Use models to predict on Unlabeled Data
            X_all_unlabeled = torch.from_numpy(no_label_data).float().unsqueeze(1)
            unlabeled_loader = DataLoader(TensorDataset(X_all_unlabeled), batch_size=BATCH_SIZE, shuffle=False)
            
            probs_u_cnn = []
            probs_u_lstm = []
            
            with torch.no_grad():
                for (x,) in tqdm(unlabeled_loader, desc=f"Pseudolabeling R{r_idx}", leave=False):
                    x = x.to(device)
                    probs_u_cnn.extend(model_cnn(x).view(-1).cpu().numpy())
                    probs_u_lstm.extend(model_lstm(x).view(-1).cpu().numpy())
            
            probs_u_cnn = np.array(probs_u_cnn)
            probs_u_lstm = np.array(probs_u_lstm)
            
            # Define Thresholds
            TH_HIGH = 0.95
            TH_LOW = 0.05
            
            # Determine Source for Next Round Data
            if r_idx < 2:
                # Self-Training: Each model learns from its OWN high predictions
                print(f"   [Strategy Round {r_idx}] Self-Training (Own Data)")
                probs_for_next_lstm = probs_u_cnn
                probs_for_next_cnn  = probs_u_cnn
                
                source_name_lstm = "LSTM (Self)"
                source_name_cnn  = "CNN (Self)"
            else:
                # # Interactive: Cross-Feeding
                # print(f"   [Strategy Round {r_idx}] Interactive Co-Training (Cross Data)")
                # probs_for_next_lstm = probs_u_cnn
                # probs_for_next_cnn  = probs_u_lstm
                
                # source_name_lstm = "CNN (Cross)"
                # source_name_cnn  = "LSTM (Cross)"

                # 优化点1：从粗暴的交叉喂数据改为"双模型共识" (Ensemble Consensus)。
                # 只有当两个模型都认同某样本，平均概率才会突破 TH_HIGH 或 TH_LOW。
                # 这样可以有效过滤掉某一个模型过度自信产生的错误错标。
                print(f"   [Strategy Round {r_idx}] Interactive Co-Training (Ensemble Consensus)")
                probs_ens = (probs_u_cnn + probs_u_lstm) / 2.0
                probs_for_next_lstm = probs_ens
                probs_for_next_cnn  = probs_ens
                
                source_name_lstm = "Ensemble Consensus"
                source_name_cnn  = "Ensemble Consensus"
            
            # ------------------------------------------------------------------------
            # 1. Update LSTM Training Data (using probs_for_next_lstm)
            # ------------------------------------------------------------------------
            src_afib_idx = np.where(probs_for_next_lstm > TH_HIGH)[0]
            src_normal_idx = np.where(probs_for_next_lstm < TH_LOW)[0]

            count_for_lstm = min(len(src_afib_idx), len(src_normal_idx), 2500)
            
            if count_for_lstm > 0:
                # Top K
                top_afib = np.argsort(probs_for_next_lstm[src_afib_idx])[::-1][:count_for_lstm]
                idx_afib_for_lstm = src_afib_idx[top_afib]
                
                top_normal = np.argsort(probs_for_next_lstm[src_normal_idx])[:count_for_lstm]
                idx_normal_for_lstm = src_normal_idx[top_normal]
                
                # Construct training set for NEXT round LSTM
                X_af = no_label_data[idx_afib_for_lstm]
                X_nm = no_label_data[idx_normal_for_lstm]
                curr_X_train_lstm = np.concatenate([X_af, X_nm])
                curr_y_train_lstm = np.concatenate([np.ones(len(X_af)), np.zeros(len(X_nm))])
                # Shuffle
                p = np.random.permutation(len(curr_X_train_lstm))
                curr_X_train_lstm, curr_y_train_lstm = curr_X_train_lstm[p], curr_y_train_lstm[p]
                
                print(f"   Next LSTM Data: {len(curr_X_train_lstm)} (Source: {source_name_lstm})")
            else:
                print(f"   Warning: Source ({source_name_lstm}) has no confident samples. Keeping LSTM data same.")

            # ------------------------------------------------------------------------
            # 2. Update CNN Training Data (using probs_for_next_cnn)
            # ------------------------------------------------------------------------
            src_afib_idx = np.where(probs_for_next_cnn > TH_HIGH)[0]
            src_normal_idx = np.where(probs_for_next_cnn < TH_LOW)[0]
            
            count_for_cnn = min(len(src_afib_idx), len(src_normal_idx), 2500)
            
            if count_for_cnn > 0:
                # Top K
                top_afib = np.argsort(probs_for_next_cnn[src_afib_idx])[::-1][:count_for_cnn]
                idx_afib_for_cnn = src_afib_idx[top_afib]
                
                top_normal = np.argsort(probs_for_next_cnn[src_normal_idx])[:count_for_cnn]
                idx_normal_for_cnn = src_normal_idx[top_normal]
                
                # Construct
                X_af = no_label_data[idx_afib_for_cnn]
                X_nm = no_label_data[idx_normal_for_cnn]
                curr_X_train_cnn = np.concatenate([X_af, X_nm])
                curr_y_train_cnn = np.concatenate([np.ones(len(X_af)), np.zeros(len(X_nm))])
                # Shuffle
                p = np.random.permutation(len(curr_X_train_cnn))
                curr_X_train_cnn, curr_y_train_cnn = curr_X_train_cnn[p], curr_y_train_cnn[p]
                
                print(f"   Next CNN Data: {len(curr_X_train_cnn)} (Source: {source_name_cnn})")
            else:
                 print(f"   Warning: Source ({source_name_cnn}) has no confident samples. Keeping CNN data same.")

        results_history.append(run_res)

    # 【保存】最终结果汇总
    final_results = {
        'num_runs': NUM_RUNS,
        'total_rounds': TOTAL_ROUNDS,
        'runs': results_history,
        'config': {
            'CV_TH': CV_TH,
            'ARI_TH': ARI_TH,
            'BATCH_SIZE': BATCH_SIZE,
            'NUM_EPOCHS': NUM_EPOCHS,
            'LR': LR,
            'TH_HIGH': TH_HIGH,
            'TH_LOW': TH_LOW
        }
    }
    
    with open(os.path.join(INTERMEDIATE_DIR, "final_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)

    print("\n✓ 完成！所有中间数据已保存至 './intermediate_results/' 目录")
    print("  - initial_pseudo_labels.npz: 初始规则生成的伪标签")
    print("  - test_set.npz: 测试集数据")
    print("  - run_X/round_Y/: 每轮迭代的详细数据")
    print("  - final_results.json: 最终汇总结果")

    print("\nDone. Interactive Training Complete.")
