import os
import numpy as np
from data_processing.Dealdata import ECG_Datadeal
from scipy.signal import filtfilt, butter, find_peaks
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


if __name__ == '__main__':
    print("This is the main script for self-supervised learning.")
    print(">>> 开始加载和预处理数据...")
    train_npy_path = ECG_Datadeal(os.path.join('data', 'train', 'traindata.mat'))
    all_data = np.load(train_npy_path)
    label_data = all_data[:1000]
    no_label_data = all_data[1000:]

    print(">>> 有标签数据加载完成，数据形状:", label_data.shape)
    print(">>> 无标签数据加载完成，数据形状:", no_label_data.shape)

    # 使用传统学习方法进行特征提取和分类
    print(">>> 使用传统方法进行特征提取和分类...")

    # 标记数据标签 (假设后续数据为正常心电)

    # CNN训练数据集

    # 验证效果