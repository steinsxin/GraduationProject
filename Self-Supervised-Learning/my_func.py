import numpy as np
import torch
import os

# ----------------------
# 模型训练函数（支持保存最优模型 & MixUp）
# ----------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, save_best=False, save_path=None, use_mixup=False, mixup_alpha=0.2):

    """
    训练模型并可选地保存验证集上表现最好的模型。
    """
    train_epochs_loss = []
    valid_epochs_loss = []
    train_epochs_acc = []
    valid_epochs_acc = []

    # 用于跟踪最优模型
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = []
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).view(-1, 1).to(device) 

            # --- MixUp Logic ---
            if use_mixup and inputs.size(0) > 1:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                index = torch.randperm(inputs.size(0)).to(device)
                
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                y_a, y_b = labels, labels[index]
                
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                
                preds = (outputs >= 0.5).float()
                target_approx = y_a if lam > 0.5 else y_b
                train_correct += (preds == target_approx).sum().item()
            
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = (outputs >= 0.5).float()
                train_correct += (preds == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_total += labels.size(0)
        
        epoch_train_loss = np.mean(train_loss)
        epoch_train_acc = train_correct / train_total
        train_epochs_loss.append(epoch_train_loss)
        train_epochs_acc.append(epoch_train_acc)

        # ========== 验证阶段 ==========
        model.eval()
        valid_loss = []
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(torch.float32).to(device)
                labels = labels.to(torch.float32).view(-1, 1).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss.append(loss.item())

                preds = (outputs >= 0.5).float()
                valid_correct += (preds == labels).sum().item()
                valid_total += labels.size(0)

        epoch_val_loss = np.mean(valid_loss)
        epoch_val_acc = valid_correct / valid_total
        valid_epochs_loss.append(epoch_val_loss)
        valid_epochs_acc.append(epoch_val_acc)

        # ========== 保存最优模型 ==========
        if save_best and save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir != "" and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                torch.save(model.state_dict(), save_path)
                # print(f"Epoch {epoch+1}: validation loss improved to {best_loss:.4f}, saving model to {save_path}")

        # ========== 学习率调整 (Old Style) ==========
        # 如果 epoch 增加到了 60，原来的 decay 太快了，稍微放缓一点
        lr_adjust = {
            10: 5e-4, 20: 1e-4, 30: 5e-5, 40: 1e-5, 50: 1e-6
        }
        if epoch in lr_adjust:
            new_lr = lr_adjust[epoch]
            for g in optimizer.param_groups:
                g['lr'] = new_lr
            print(f"Updating learning rate to {new_lr}")

        # 打印
        print(
            f"Epoch [{epoch+1}/{num_epochs}] => "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc*100:.2f}%"
        )

    return {
        'train_loss': train_epochs_loss,
        'valid_loss': valid_epochs_loss,
        'train_acc': train_epochs_acc,
        'valid_acc': valid_epochs_acc
    }

