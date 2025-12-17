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
from sklearn.model_selection import StratifiedShuffleSplit
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
    num_splits = 5  # Number of splits for each ratio, similar to k-folds

    # 定义训练比例
    split_ratios = [(4, 1), (3, 2), (2, 3), (1, 4), (1, 9), (1, 19), (1, 99)]

    # 构建数据集
    dataset = ECG_Dataset(trainset_path, mode="train", labeled_only=True)

    # 进一步打乱数据和标签，确保AF和非AF样本充分混合
    permutation = np.random.permutation(len(dataset))
    dataset.data = dataset.data[permutation]
    dataset.labels = dataset.labels[permutation]

    # 存储所有比例的最终结果
    all_ratios_results = {}

    # 外层循环：遍历不同的分割比例
    for train_ratio, val_ratio in split_ratios:
        ratio_str = f"{train_ratio}_{val_ratio}"
        val_size = val_ratio / (train_ratio + val_ratio)

        print(f"--- Starting {num_splits}-split CV for ratio {train_ratio}:{val_ratio} ---")

        # 使用StratifiedShuffleSplit进行多次切分
        sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=val_size, random_state=42)

        # 存储当前比例下每次折叠的结果
        current_ratio_results = {}
        all_folds_val_acc = []

        # 内层循环：交叉验证
        for fold, (train_ids, val_ids) in enumerate(sss.split(np.zeros(len(dataset)), dataset.labels)):
            print(f'  FOLD {fold + 1}/{num_splits}')
            print('  --------------------------------')

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
            save_path = os.path.join("save", f"Bestmodel_CNN_ratio_{ratio_str}_fold_{fold+1}.pth")
            os.makedirs("save", exist_ok=True)

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
            current_ratio_results[f'fold_{fold+1}'] = result
            all_folds_val_acc.append(max(result['valid_acc']))

        # 保存当前比例的所有折叠结果和平均性能
        all_ratios_results[f'ratio_{ratio_str}'] = {
            'details': current_ratio_results,
            'average_validation_accuracy': np.mean(all_folds_val_acc),
            'std_dev_validation_accuracy': np.std(all_folds_val_acc)
        }
        print(f"  Average Validation Accuracy for ratio {train_ratio}:{val_ratio}: {np.mean(all_folds_val_acc) * 100:.2f}% (+/- {np.std(all_folds_val_acc) * 100:.2f}%)")
        print(f"--- Finished CV for ratio {train_ratio}:{val_ratio} ---\n")

    # 保存所有比例的最终结果到json文件
    results_save_path = os.path.join("results", "ratio_cv_results.json")
    os.makedirs("results", exist_ok=True)
    with open(results_save_path, 'w') as f:
        json.dump(all_ratios_results, f, indent=4)

    print('--------------------------------')
    print('Overall Results Across All Ratios')
    for ratio_str, data in all_ratios_results.items():
        ratio = ratio_str.replace('ratio_', '').replace('_', ':')
        avg_acc = data['average_validation_accuracy']
        std_acc = data['std_dev_validation_accuracy']
        print(f"Ratio {ratio}: Average Validation Accuracy = {avg_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)")
    print(f'Overall results saved to {results_save_path}')

    # 可视化每个比例的平均训练过程
    plt.figure(figsize=(12, 5))
    
    # 绘制每个比例的平均验证准确率
    plt.subplot(1, 2, 1)
    for ratio_str, data in all_ratios_results.items():
        ratio = ratio_str.replace('ratio_', '').replace('_', ':')
        # Calculate average accuracy curve across folds
        avg_val_acc = np.mean([d['valid_acc'] for d in data['details'].values()], axis=0)
        plt.plot(range(1, num_epochs + 1), avg_val_acc, label=f'Avg Val Acc Ratio {ratio}')
    
    plt.title('Average Validation Accuracy per Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.legend()

    # 绘制每个比例的平均验证损失
    plt.subplot(1, 2, 2)
    for ratio_str, data in all_ratios_results.items():
        ratio = ratio_str.replace('ratio_', '').replace('_', ':')
        # Calculate average loss curve across folds
        avg_val_loss = np.mean([d['valid_loss'] for d in data['details'].values()], axis=0)
        plt.plot(range(1, num_epochs + 1), avg_val_loss, label=f'Avg Val Loss Ratio {ratio}')

    plt.title('Average Validation Loss per Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()

    plt.tight_layout()
    # 保存图像
    plot_save_path = os.path.join("results", "ratio_cv_metrics.png")
    plt.savefig(plot_save_path)
    print(f"Plots saved to {plot_save_path}")
    plt.show()
