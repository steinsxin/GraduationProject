# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.signal import filtfilt, butter, find_peaks
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# ---------- 1. 全局参数 ----------
FS = 400
N_JOBS = -1

# ---------- 2. 核心信号处理函数 ----------
def _bandpass(sig, fs=FS, low=5, high=45):
    """带通滤波，保留QRS波主要频段"""
    b, a = butter(3, [low, high], btype='band', fs=fs)
    return filtfilt(b, a, sig, padlen=0)

def _detect_qrs(sig, fs=FS):
    """简化版Pan-Tompkins QRS检测"""
    win = int(0.2 * fs)
    diff = np.r_[0, 0, sig[2:] - sig[:-2]]
    
    # 动态阈值计算
    positive_diff = diff[diff > 0]
    if len(positive_diff) == 0: return np.array([]) # 避免在平坦信号上出错
    DDpos = np.percentile(positive_diff, 75)
    threshold = max(DDpos * 1.2, np.percentile(np.abs(sig), 75) * 0.4)
    
    peaks, _ = find_peaks(diff, height=threshold, distance=win)
    return peaks

def _calculate_features(ecg):
    """从单段ECG计算CV和ARI特征"""
    sig = _bandpass(ecg, FS)
    r_peaks = _detect_qrs(sig, FS)
    rr = np.diff(r_peaks) / FS
    
    # 确保有足够的RR间期进行可靠计算
    if len(rr) < 5:
        return 0, 0
        
    mean_rr = np.mean(rr)
    if mean_rr == 0: return 0,0 # 避免除零错误

    cv = np.std(rr) / mean_rr
    sum_rr = np.sum(rr)
    if sum_rr == 0: return cv, 0 # 避免除零错误
    
    ari = np.sum(np.abs(np.diff(rr))) / sum_rr
    return cv, ari

# ---------- 3. 阈值寻优与评估 ----------
def find_optimal_thresholds(features, labels):
    """
    在训练集上通过网格搜索寻找最佳阈值。
    """
    best_f1 = -1
    best_thresholds = (0, 0)
    
    # 定义阈值搜索范围
    cv_range = np.arange(0.02, 0.20, 0.01)
    ari_range = np.arange(0.10, 0.40, 0.02)

    for cv_th in cv_range:
        for ari_th in ari_range:
            # 应用阈值进行预测
            predictions = (features[:, 0] > cv_th) & (features[:, 1] > ari_th)
            score = f1_score(labels, predictions, zero_division=0)
            
            if score > best_f1:
                best_f1 = score
                best_thresholds = (cv_th, ari_th)
                
    return best_thresholds

# ---------- 4. 主执行逻辑：5折交叉验证 ----------
if __name__ == '__main__':
    # 1. 数据加载与预处理
    from data_processing.Dealdata import ECG_Datadeal
    print(">>> 步骤1/4: 开始加载和预处理数据...")
    train_npy_path = ECG_Datadeal(os.path.join('data', 'train', 'traindata.mat'))
    all_data = np.load(train_npy_path)

    # 2. 准备数据集 (前1000条, 500 AF + 500 Normal)
    print(">>> 步骤2/4: 准备数据集和标签...")
    data_to_use = all_data[:1000]
    labels = np.array([1] * 500 + [0] * 500)
    
    # 创建索引并打乱，确保数据和标签对应关系
    indices = np.arange(len(data_to_use))
    np.random.shuffle(indices)
    data_to_use = data_to_use[indices]
    labels = labels[indices]

    # 3. 并行计算所有样本的特征 (一次性完成，避免重复计算)
    print(">>> 步骤3/4: 并行计算所有样本的CV和ARI特征...")
    feature_list = Parallel(n_jobs=N_JOBS)(
        delayed(_calculate_features)(data_to_use[i]) for i in tqdm(range(len(data_to_use)))
    )
    features = np.array(feature_list)

    # 4. 执行5折交叉验证
    print(">>> 步骤4/4: 开始5折交叉验证...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for i, (train_idx, val_idx) in enumerate(kf.split(features)):
        print(f"\n----- 第 {i+1}/5 折 -----")
        
        # 划分训练集和验证集
        train_features, val_features = features[train_idx], features[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        
        # 在训练集上寻找最优阈值
        print("  - 在训练集上搜索最佳阈值...")
        best_cv_th, best_ari_th = find_optimal_thresholds(train_features, train_labels)
        print(f"  - 找到最佳阈值: CV_TH={best_cv_th:.2f}, ARI_TH={best_ari_th:.2f}")
        
        # 在验证集上评估模型
        predictions = (val_features[:, 0] > best_cv_th) & (val_features[:, 1] > best_ari_th)
        accuracy = accuracy_score(val_labels, predictions)
        fold_accuracies.append(accuracy)
        print(f"  - 验证集准确率: {accuracy:.4f}")

    # 5. 打印最终结果
    mean_accuracy = np.mean(fold_accuracies)
    print("\n==========================================")
    print(f"5折交叉验证完成")
    print(f"各折准确率: {[round(acc, 4) for acc in fold_accuracies]}")
    print(f"平均准确率: {mean_accuracy:.4f}")
    print("==========================================")