# -*- coding: UTF-8 -*-
# ----------------------
# 导入需要的包
# ----------------------
"""第三方库"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import json

"""自定义模块"""
import my_func
from model.CNN import CNN
from data_processing.Dealdata import ECG_Datadeal
from data_processing.Dataset import ECG_Dataset

# ----------------------
# 数据准备与预处理
# ----------------------

# 使用 os.path 自动处理路径分隔符
train_path = os.path.join("data", "train", "traindata.mat")
test_path = os.path.join("data", "test", "testdata.mat")

# 预处理数据并返回保存路径（.npy）
trainset_path = ECG_Datadeal(train_path)
testset_path  = ECG_Datadeal(test_path)

# ----------------------
# 主程序入口
# ----------------------

if __name__ == "__main__":
    # 配置训练参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    num_epochs = 40
    batch_size = 8
    learning_rate = 0.001
    k_folds = 5

    # 构建数据集
    dataset = ECG_Dataset(trainset_path, mode="train", labeled_only=True)

    # 进一步打乱数据和标签，确保AF和非AF样本充分混合
    permutation = np.random.permutation(len(dataset))
    dataset.data = dataset.data[permutation]
    dataset.labels = dataset.labels[permutation]

    # 定义K-fold交叉验证
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # 存储每次折叠的结果
    results = {}
    all_folds_val_acc = []

    # K-fold交叉验证循环
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold+1}/{k_folds}')
        print('--------------------------------')

        # 创建数据采样器和加载器
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        # 初始化模型
        model = CNN().to(device)

        # 定义损失函数与优化器
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 可选：保存最佳模型
        save_path = os.path.join("save", f"Bestmodel_CNN_fold_{fold+1}.pth")

        # 训练模型
        result = my_func.train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device,
            save_best=True,
            save_path=save_path
        )

        # 保存该折叠的结果
        results[f'fold_{fold+1}'] = {
            'train_loss': result['train_loss'],
            'valid_loss': result['valid_loss'],
            'train_acc': result['train_acc'],
            'valid_acc': result['valid_acc']
        }
        all_folds_val_acc.append(max(result['valid_acc']))

    # 保存所有折叠的结果到json文件
    results_save_path = os.path.join("results", "5_fold_cv_results.json")
    os.makedirs("results", exist_ok=True)
    with open(results_save_path, 'w') as f:
        json.dump(results, f, indent=4)

    print('--------------------------------')
    print(f'K-fold cross-validation results for {k_folds} folds')
    print(f'Average Validation Accuracy: {np.mean(all_folds_val_acc) * 100:.2f}% (+/- {np.std(all_folds_val_acc) * 100:.2f}%)')
    print(f'Results saved to {results_save_path}')

    # 可视化训练过程
    plt.figure(figsize=(12, 5))

    # 绘制每个折叠的验证准确率
    plt.subplot(1, 2, 1)
    for fold in results:
        plt.plot(range(1, num_epochs + 1), results[fold]['valid_acc'], label=f'Val Acc Fold {fold.split("_")[-1]}')
    plt.title('Validation Accuracy per Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制每个折叠的验证损失
    plt.subplot(1, 2, 2)
    for fold in results:
        plt.plot(range(1, num_epochs + 1), results[fold]['valid_loss'], label=f'Val Loss Fold {fold.split("_")[-1]}')
    plt.title('Validation Loss per Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    # Save the figure
    plot_save_path = os.path.join("results", "5_fold_cv_metrics.png")
    plt.savefig(plot_save_path)
    print(f"Plots saved to {plot_save_path}")
    plt.show()