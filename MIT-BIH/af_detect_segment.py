# -*- coding: utf-8 -*-
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter, find_peaks
from joblib import Parallel, delayed

# ---------- 1. 固定经验阈值（400 Hz） ----------
FS = 400
CV_TH = 0.07
ARI_TH = 0.30

# ---------- 2. 信号处理 & 检峰 ----------
def _bandpass(sig, fs=FS, low=5, high=45):
    b, a = butter(3, [low, high], btype='band', fs=fs)
    return filtfilt(b, a, sig)

def _detect_qrs(sig, fs=FS):
    win = int(0.2 * fs)
    diff = np.r_[0, 0, sig[2:] - sig[:-2]]
    DDpos = np.percentile(diff[diff > 0], 75)
    threshold = max(DDpos * 1.2, np.percentile(np.abs(sig), 75) * 0.4)
    peaks, _ = find_peaks(diff, height=threshold, distance=win)
    return peaks

# ---------- 3. 单段检测（无标签版） ----------
def _single_detect(ecg):
    sig = _bandpass(ecg, FS)
    r = _detect_qrs(sig, FS)
    rr = np.diff(r) / FS
    if len(rr) < 5:
        return False
    cv = np.std(rr) / np.mean(rr)
    ari = np.sum(np.abs(np.diff(rr))) / np.sum(rr)
    return (cv > CV_TH) and (ari > ARI_TH)

# ---------- 4. 并行打标签（前 max_seg 段） ----------
def tag_npy_file(npy_path, save_plot=True, max_seg=1000, n_jobs=-1):
    os.makedirs('af_plots', exist_ok=True)
    data = np.load(npy_path)          # (N,4000)

    def _worker(idx):
        flag = _single_detect(data[idx])
        label = 'AF' if flag else 'Normal'
        if save_plot:
            plt.figure(figsize=(8, 2.5))
            plt.plot(data[idx], lw=0.6)
            plt.title(f'{label} – seg{idx}'); plt.grid(True, ls='--', alpha=0.3)
            plt.savefig(f'af_plots/seg{idx}.png', dpi=120, bbox_inches='tight')
            plt.close()
        return idx, label

    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_worker)(i) for i in range(min(max_seg, data.shape[0]))
    )
    return results

# ---------- 5. 纯计数 ----------
def count_af(npy_path, max_seg=None):
    data = np.load(npy_path)
    total = data.shape[0] if max_seg is None else min(max_seg, data.shape[0])
    af_cnt = sum(Parallel(n_jobs=-1)(delayed(_single_detect)(data[i]) for i in range(total)))
    print(f'[COUNT] 总段数:{total}  被判AF:{af_cnt}  占比:{af_cnt/total*100:.2f}%')

# ---------- 6. 主入口 ----------
if __name__ == '__main__':
    from data_processing.Dealdata import ECG_Datadeal
    train_npy = ECG_Datadeal(os.path.join('data', 'train', 'traindata.mat'))

    # ① 打标签 + 画图（前 1000）
    train_results = tag_npy_file(train_npy, save_plot=True, max_seg=10)
    np.savetxt('traditional_af_train.csv', train_results,
               fmt='%s', delimiter='\t', header='idx\tlabel', comments='')
    print('>>> 无标签版完成，标签已写入 traditional_af_train.csv')

    # ② 可选：整集纯计数（无图，秒级）
    # count_af(train_npy, max_seg=None)          # 把 20 000 段全部跑一遍