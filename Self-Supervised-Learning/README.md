# Iterative Self-Supervised Learning for ECG AF Detection

本项目实现了一种**迭代式自监督学习 (Iterative Self-Supervised Learning)** 方法，旨在**完全仅利用无标签 ECG 数据**训练出一个高精度的房颤 (AFib) 检测模型。

## 核心理念

传统的监督学习依赖大量人工标注的数据，但在医疗领域获取标注往往成本高昂。本方法采用 **"Cold Start + Self-Training"** 的策略：
1.  利用医学先验知识（专家规则）进行“冷启动”，生成初始伪标签。
2.  利用深度学习模型进行“自我提炼”，进一步发掘数据中的潜在特征并修正规则的偏差。
3.  **严格零标签泄露**：真实的有标签数据（Labeled Data）被严格隔离，仅作为最终的测试集（Test Set）来评估算法性能，绝不参与任何训练或验证环节。

---

## 算法流程 (Workflow)

整个流程分为两个主要的迭代阶段 (Round 1 & Round 2)：

### Part 1: Rule-based Cold Start (基于规则的冷启动)
在没有任何模型的情况下，我们首先利用心脏电生理学的先验知识来生成第一批训练数据。

1.  **特征提取**: 对无标签信号提取 Lorenz Plot 相关的形态学特征：
    *   **CV (Coefficient of Variation)**: RR 间期的变异系数。
    *   **ARI (Average RR Interval)**: 平均 RR 间期变化率。
2.  **专家规则判定**:
    *   基于 `CV` 和 `ARI` 设定严格的医学阈值。
    *   计算每个样本属于 AFib 或 Normal 的置信度得分。
3.  **高质量样本筛选 (Quality Control)**:
    *   **Strict Thresholding**: 仅保留置信度 > 85% 的样本。
    *   **Top-K Balancing**: 强制让 AFib 和 Normal 类别数量一致（基于较少类的数量），并仅截取置信度最高的 Top-K 个样本。
    *   *目的：宁缺毋滥，构建一个规模较小但极为纯净的初始训练集。*

### Part 2: Round 1 Training (第一轮训练)
*   使用 Part 1 生成的高质量伪标签数据，训练第一个 CNN 模型 (`CNN_Step1_RuleBased`)。
*   该模型学习到了规则所定义的特征模式，并开始具备一定的泛化能力。

### Part 3: Model-based Refinement (模型驱动的自我提炼)
规则虽然准确但往往过于保守（Recall 较低），且无法覆盖所有复杂的波形变异。我们在这一步利用训练好的 CNN 来“扩充”和“清洗”数据集。

1.  **全量推断**: 使用 `CNN_Step1` 模型对**所有**无标签数据进行预测。
2.  **高置信度重采样**:
    *   选取模型预测概率极高 (> 0.95) 或极低 (< 0.05) 的样本。
    *   *优势*：这一步能找出那些“规则认为模棱两可，但 CNN 觉得特征很典型”的样本，从而显著扩充训练数据量。同时，CNN 也能剔除部分符合规则但波形充满噪声的异常数据。
3.  **二次平衡**: 再次执行 Top-K 平衡策略，生成第二代训练集 (`X_train_r2`)。

### Part 4: Round 2 Training (最终训练)
*   使用经过模型提炼后的数据集 (`X_train_r2`) 训练最终的 CNN 模型 (`CNN_Final_iter2`)。
*   这个模型通常比第一轮模型具有更强的鲁棒性和更高的准确率。

---

## 数据使用策略 (Data Usage Protocol)

为了保证实验的严谨性，我们严格遵守以下数据划分：

| 数据集 | 来源 | 用途 | 是否包含真实标签 |
| :--- | :--- | :--- | :--- |
| **no_label_data** | 原始无标签库 | 生成伪标签、模型训练、模型验证 | **否** (完全无监督) |
| **X_train (Pseudo)** | 从无标签库筛选 | 训练模型参数 | 否 (使用伪标签) |
| **X_val (Pseudo)** | 从无标签库筛选 | Early Stopping, 监控过拟合 | 否 (使用伪标签) |
| **label_data** | 独立的有标签库 | **仅用于最终测试 (Final Test)** | **是** (仅用于评估) |

## 关键技术点

*   **Iterative Self-Training**: 从“规则”到“模型”的知识蒸馏过程。
*   **Dynamic Top-K Selection**: 动态调整每一轮的数据选择数量，确保训练集始终处于“类别平衡”且“置信度最优”的状态。
*   **Pseudo-Label Validation**: 训练过程中的验证集也是从伪标签数据中划分的（80% 训练 / 20% 验证），这确保了模型选择不依赖任何外部标签信息。

## 运行方式

```bash
python main.py
```
程序将自动执行上述所有步骤，并在最后输出最终模型在真实标签测试集上的 Accuracy 和 F1-Score。
