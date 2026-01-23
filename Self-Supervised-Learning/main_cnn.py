import os
import numpy as np
from data_processing.Dealdata import ECG_Datadeal
from scipy.signal import filtfilt, butter, find_peaks
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import json
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import my_func
from model.CNN import CNN

# ============================================================
# 全局参数
# ============================================================
BATCH_SIZE = 32
NUM_EPOCHS = 40
LR = 1e-3

def make_loader(X, y, shuffle):
    X = torch.from_numpy(X).float().unsqueeze(1)
    y = torch.from_numpy(y).float()
    return DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n================ Part 0: 数据加载 =================")

    train_npy_path = ECG_Datadeal(os.path.join("data", "train", "traindata.mat"))
    all_data = np.load(train_npy_path)

    label_data = all_data[:1000]
    no_label_data = all_data[1000:]

    labels = np.array([1] * 500 + [0] * 500)

    print(f"有标签数据: {label_data.shape}")
    print(f"无标签数据: {no_label_data.shape}")

    # 所有有标签数据作为最终测试集
    X_test = label_data
    y_test = labels
    print(f"Test Set (All Labeled Data): {X_test.shape}")

    # 获取所有权重文件
    weight_files = glob.glob(os.path.join("weights", "*.pth"))
    
    def sort_key(f):
        try:
            parts = os.path.basename(f).split('_')
            if 'ratio' in parts:
                idx = parts.index('ratio')
                return int(parts[idx+2])
        except:
            pass
        return 0

    weight_files.sort(key=sort_key, reverse=True)
    
    if not weight_files:
        print("No weight files found in 'weights/' directory.")
        exit()

    print(f"Found {len(weight_files)} models to process: {[os.path.basename(f) for f in weight_files]}")

    final_results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for weights_path in weight_files:
        model_name = os.path.basename(weights_path).replace(".pth", "")
        print(f"\n\n{'='*40}")
        print(f"Processing Model: {model_name}")
        print(f"{'='*40}")

        # ========================================================
        # Part 1：无标签数据 → 伪标签
        # ========================================================
        print(f"Loading weights from: {weights_path}")

        pretrained_model = CNN().to(device)
        pretrained_model.load_state_dict(torch.load(weights_path, map_location=device))
        pretrained_model.eval()

        # Create DataLoader for unlabeled data
        X_unlabeled_tensor = torch.from_numpy(no_label_data).float().unsqueeze(1)
        unlabeled_loader = DataLoader(TensorDataset(X_unlabeled_tensor), batch_size=BATCH_SIZE, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for (x,) in tqdm(unlabeled_loader, desc="CNN Inference"):
                x = x.to(device)
                out = pretrained_model(x).view(-1)
                all_probs.extend(out.cpu().numpy())

        all_probs = np.array(all_probs)

        # 筛选高置信度样本
        AFIB_TH = 0.55
        NORMAL_TH = 0.2

        afib_idx = np.where(all_probs > AFIB_TH)[0]
        normal_idx = np.where(all_probs < NORMAL_TH)[0]

        X_afib = no_label_data[afib_idx]
        X_normal = no_label_data[normal_idx]

        print(f"伪 AFib (Prob > {AFIB_TH}): {len(X_afib)}")
        print(f"伪 Normal (Prob < {NORMAL_TH}): {len(X_normal)}")

        if len(X_afib) == 0 or len(X_normal) == 0:
            print(f"⚠️ Model {model_name}: 伪标签严重失衡，跳过此模型或降低阈值")
            # 可以选择跳过，或者继续但效果可能不好。这里记录结果为失败或0
            final_results[model_name] = {
                "accuracy": 0.0,
                "f1_score": 0.0,
                "status": "failed_imbalanced"
            }
            continue

        X_train = np.concatenate([X_afib, X_normal])
        y_train = np.concatenate([
            np.ones(len(X_afib)),
            np.zeros(len(X_normal))
        ])

        perm = np.random.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        # ========================================================
        # Part 3：PyTorch 训练 (Semi-Supervised)
        # ========================================================
        print(f"仅使用伪标签数据进行训练和验证划分 ({X_train.shape[0]})...")
        
        # 将伪标签数据划分为 训练集(80%) 和 验证集(20%)
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=42
        )
        
        print(f"Training Set (Pseudo): {X_train_final.shape}")
        print(f"Validation Set (Pseudo): {X_val.shape}")

        train_loader = make_loader(X_train_final, y_train_final, shuffle=True)
        val_loader = make_loader(X_val, y_val, False)

        model = CNN().to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        save_dir = "save"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"Bestmodel_SemiSupervised_{model_name}.pth")

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

        # ========================================================
        # Part 4: 保存训练过程图
        # ========================================================
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['valid_loss'], label='Validation Loss')
        plt.title(f'Loss Curve ({model_name})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['valid_acc'], label='Validation Accuracy')
        plt.title(f'Accuracy Curve ({model_name})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        results_dir = os.path.join(base_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        save_fig_path = os.path.join(results_dir, f"training_process_{model_name}.png")
        plt.savefig(save_fig_path)
        plt.close() # Close plot to free memory

        # ========================================================
        # Part 5：最终测试
        # ========================================================
        state_dict = torch.load(save_path, weights_only=True)
        model.load_state_dict(state_dict)

        test_loader = make_loader(X_test, y_test, False)

        preds, gts = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = model(x).squeeze()
                pred = (out > 0.5).int().cpu().numpy()
                preds.extend(pred)
                gts.extend(y.numpy())

        acc = accuracy_score(gts, preds)
        f1 = f1_score(gts, preds)

        print(f"Model {model_name} Result -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

        final_results[model_name] = {
            "accuracy": acc,
            "f1_score": f1,
            "pseudo_afib_count": int(len(X_afib)),
            "pseudo_normal_count": int(len(X_normal)),
            "status": "success"
        }

    # ========================================================
    # Part 6: 汇总对比 & 保存结果
    # ========================================================
    print("\n================ Part 6: Summary & Plotting =================")
    
    # Save JSON
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "all_models_comparison.json")
    with open(json_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"All results saved to {json_path}")

    # Plot Comparison
    valid_results = {k: v for k, v in final_results.items() if v.get("status") == "success"}
    
    if valid_results:
        names = list(valid_results.keys())
        # Shorten names for plotting if too long
        short_names = [n.replace("Bestmodel_CNN_", "").replace("_fold_1", "") for n in names]
        
        accs = [valid_results[n]['accuracy'] for n in names]
        f1s = [valid_results[n]['f1_score'] for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        plt.figure(figsize=(max(10, len(names)*1.5), 6))
        rects1 = plt.bar(x - width/2, accs, width, label='Accuracy')
        rects2 = plt.bar(x + width/2, f1s, width, label='F1 Score')
        
        plt.ylabel('Scores')
        plt.title('Semi-Supervised Learning Performance Comparison')
        plt.xticks(x, short_names, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.ylim(0, 1.1)
        
        # Add labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90)

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        save_plot_path = os.path.join(results_dir, "all_models_comparison.png")
        plt.savefig(save_plot_path)
        print(f"Comparison plot saved to {save_plot_path}")
    else:
        print("No valid results to plot.")
