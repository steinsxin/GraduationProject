# -*- coding: UTF-8 -*-
# 测试验证数据并保存为csv文件
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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
    test_dataset = ECG_Dataset(testset_path, mode="test", labeled_only=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --------------------------
    # 加载模型
    # --------------------------
    model_path = os.path.join("save", "Bestmodel_CNN.pth")
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Model Loaded.")

    # --------------------------
    # 推理
    # --------------------------
    predictions = []
    dummy_labels = []  # test 数据 label 全为 -1

    with torch.no_grad():
        for signals, labels in test_loader:

            signals = signals.float().to(device) # shape (1,1,4000)

            outputs = model(signals)        # shape (1,1)
            pred = outputs.cpu().numpy().flatten()[0]

            predictions.append(pred)
            dummy_labels.append(labels.item())   # 全是 -1，可选

    # --------------------------
    # 保存 CSV
    # --------------------------
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "test_result.csv")

    df = pd.DataFrame({
        "prediction": predictions,
        "label": dummy_labels   # 若不需要，可删掉
    })

    df.to_csv(csv_path, index=False)
    print(f"\n测试完成，结果已保存到：{csv_path}\n")


if __name__ == "__main__":
    test()
