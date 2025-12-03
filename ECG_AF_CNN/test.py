# -*- coding: UTF-8 -*-
# 测试验证数据并保存为csv文件
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import glob

from data_processing.Dealdata import ECG_Datadeal
from data_processing.Dataset import ECG_Dataset
from model.CNN import CNN


def test():

    # --------------------------
    # 设备
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------
    # 数据处理 testdata.mat → testdata.npy
    # --------------------------
    test_mat_path = os.path.join("data", "test", "testdata.mat")
    testset_path = ECG_Datadeal(test_mat_path)

    # --------------------------
    # 构造测试集
    # --------------------------
    test_dataset = ECG_Dataset(testset_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --------------------------
    # 加载模型
    # --------------------------
    model_paths = sorted(glob.glob(os.path.join("save", "Bestmodel_CNN_fold_*.pth")))
    if not model_paths:
        print("No fold models found. Please run train.py with cross-validation first.")
        return

    print(f"Found {len(model_paths)} models for ensembling.")

    all_predictions = []
    dummy_labels = []  # test 数据 label 全为 -1

    # --------------------------
    # 推理
    # --------------------------
    for i, model_path in enumerate(model_paths):
        print(f"--- Predicting with model {i+1}/{len(model_paths)}: {os.path.basename(model_path)} ---")
        model = CNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_predictions = []
        with torch.no_grad():
            # 只在第一次迭代时收集标签
            if i == 0:
                for signals, labels in test_loader:
                    signals = signals.float().to(device)
                    outputs = model(signals)
                    pred = outputs.cpu().numpy().flatten()[0]
                    fold_predictions.append(pred)
                    dummy_labels.append(labels.item())
            else:
                for signals, _ in test_loader:
                    signals = signals.float().to(device)
                    outputs = model(signals)
                    pred = outputs.cpu().numpy().flatten()[0]
                    fold_predictions.append(pred)


        all_predictions.append(fold_predictions)

    # --------------------------
    # 集成预测
    # --------------------------
    # all_predictions 是一个 (模型数量, 样本数量) 的列表
    # 我们需要在每个样本上对模型进行平均
    ensembled_predictions = np.mean(all_predictions, axis=0)
    
    # 同样可以为最终的预测标签使用多数投票
    # 例如，对平均概率使用阈值
    final_labels = (ensembled_predictions >= 0.5).astype(int)


    # --------------------------
    # 保存 CSV
    # --------------------------
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "test_result_ensembled.csv")

    df = pd.DataFrame({
        "ensembled_prediction_prob": ensembled_predictions,
        "ensembled_prediction_label": final_labels,
        "original_label": dummy_labels
    })

    df.to_csv(csv_path, index=False)
    print(f"\n测试完成，集成结果已保存到：{csv_path}\n")


if __name__ == "__main__":
    test()
