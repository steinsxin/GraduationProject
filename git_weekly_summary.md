# Git 提交周整理

说明：本文基于 `full_history.patch` 中的提交时间、提交说明和改动路径整理，按自然周（周一至周日）归类。部分周的工作内容是根据代码改动方向做的归纳，用来回答“那一周主要干了什么、在推进什么事情”。

## 一、整体阶段划分

1. 2025 年 11 月下旬：项目初始化，打通数据读取、可视化和 MIT-BIH 相关代码骨架。
2. 2025 年 12 月：重点推进传统房颤检测、CNN 训练流程和自监督/半监督学习项目初始化。
3. 2026 年 1 月到 3 月：重心转向 Self-Supervised-Learning，持续做交互式训练、LSTM/Transformer 实验、精度修复和结果对比。
4. 2026 年 4 月到 5 月：新增 MIT-Learning 分支，复现并提升 CNN、CNN+LSTM 等模型表现，同时补充文档并收尾整理结果。

## 二、按周整理

### 第 1 周（2025-11-17 ~ 2025-11-23）

- 代表提交：`Init`、`update`、`update MatDataVisualizer`、`update MIT-BIH`
- 主要改动目录：`main.py`、`README.md`、`requirements.txt`、`ECG_AF_Net/`、`MIT-BIH/`
- 这一周在做什么：
  - 初始化整个项目仓库，补齐基础的 Python 工程文件和依赖说明。
  - 编写 `.mat` 文件读取逻辑，开始接入 CPSC2025 数据集。
  - 增加 MatDataVisualizer 一类的可视化能力，用于查看 ECG 样本和确认数据格式。
  - 引入 MIT-BIH 相关代码目录，开始搭建房颤检测的第二条实验线。
- 阶段目标：先把“数据能读、样本能看、项目能跑”的基础设施搭起来，为后续训练和算法实验做准备。

### 第 2 周（2025-12-01 ~ 2025-12-07）

- 代表提交：`update`、`paper`、`update af_detect & gitignore & README`、`refactor ECG_CNN`、`update ECG`、`refactor`
- 主要改动目录：`MIT-BIH/`、`MIT_BIH/`、`ECG_AF_CNN/`、`plots/`
- 这一周在做什么：
  - 围绕 MIT-BIH 数据集完善传统房颤检测流程，包括 `af_detect_segment.py`、训练标签导出和分段图像输出。
  - 补充 README 和论文相关材料，说明算法思路和实验背景。
  - 对 ECG CNN 训练代码做一次较大的重构，开始把“数据处理、模型定义、训练流程、结果输出”拆分清楚。
  - 输出大量样本可视化和中间结果图，说明这一周也在做数据核验和算法行为观察。
- 阶段目标：从单纯“读数据”过渡到“能做传统算法检测、能训练 CNN、能看见结果图”的阶段。

### 第 3 周（2025-12-08 ~ 2025-12-14）

- 代表提交：`fix packet path issue`、`fix data`、`Init`、`update logic & model`
- 主要改动目录：`MIT_BIH/`、`ECG_AF_CNN/`、`Self-Supervised-Learning/`
- 这一周在做什么：
  - 修复包路径和数据处理问题，让 MIT_BIH 和 CNN 训练代码在项目结构里能稳定运行。
  - 保存 `ECG_AF_CNN` 的 5 折交叉验证结果和模型权重，说明这时已经进入正式训练和结果记录阶段。
  - 新建 `Self-Supervised-Learning/` 项目，拷贝并整理数据处理模块、主程序和 CNN 模型代码。
  - 开始把研究重点从传统监督训练扩展到自监督/半监督方向。
- 阶段目标：在已有 CNN 实验基础上，开出新的自监督实验支线，并保证原来的数据与训练流程可复用。

### 第 4 周（2025-12-15 ~ 2025-12-21）

- 代表提交：`update`、`update`、`fix`
- 主要改动目录：`ECG_AF_CNN/`、`Self-Supervised-Learning/`
- 这一周在做什么：
  - 持续完善 `ECG_AF_CNN/train.py` 和实验结果输出，加入不同划分比例下的交叉验证指标。
  - 在 `Self-Supervised-Learning/` 中加入伪标签样本图、`af_detect_segment.py` 和辅助函数，开始把传统检测思路和半监督训练连接起来。
  - 从结果文件看，已经开始筛选伪标签样本、保存可视化案例，说明工作重点是“如何让未标注样本参与训练”。
- 阶段目标：构建半监督学习闭环，即先得到伪标签，再把伪标签数据纳入模型训练和效果评估。

### 第 5 周（2025-12-29 ~ 2026-01-04）

- 代表提交：`update`、`save frist`、`update`
- 主要改动目录：`Self-Supervised-Learning/`
- 这一周在做什么：
  - 在 `main_cnn.py` 中推进半监督 CNN 主实验。
  - 保存第一批半监督模型权重和训练过程图，例如 `Bestmodel_SemiSupervised_CNN.pth`。
  - 汇总不同标注比例下的模型对比结果，输出 `all_models_comparison.json/png`。
- 阶段目标：把半监督学习从“能跑”推进到“能系统比较不同数据比例和模型表现”。

### 第 6 周（2026-01-05 ~ 2026-01-11）

- 代表提交：`update`、`fix`、`fix`、`update`、`update`
- 主要改动目录：`Self-Supervised-Learning/`、`ECG_AF_CNN/`
- 这一周在做什么：
  - 继续修正半监督训练脚本中的问题，补齐不同折、不同比例的结果图和结果文件。
  - 更新 `main.py`、`main_cnn.py` 和模型保存结果，说明实验已经从单次训练转为多组对比实验。
  - 同步维护 `ECG_AF_CNN` 的比例实验结果，用于和半监督方案做横向比较。
- 阶段目标：稳定实验管线，确保每种设置都能输出完整结果，便于后面总结哪一种方案更有效。

### 第 7 周（2026-01-19 ~ 2026-01-25）

- 代表提交：`update`、`update`、`update`
- 主要改动目录：`Self-Supervised-Learning/main.py`、`Self-Supervised-Learning/model/CNN.py`、`Self-Supervised-Learning/my_func.py`
- 这一周在做什么：
  - 集中修改自监督主流程、CNN 模型和工具函数。
  - 补充 `final_stats.json` 与训练过程图，说明开始做阶段性统计和结果沉淀。
- 阶段目标：把半监督 CNN 的实现整理得更稳定，准备进入下一阶段的新模型或新训练策略实验。

### 第 8 周（2026-01-26 ~ 2026-02-01）

- 代表提交：`update`、`fix`
- 主要改动目录：`Self-Supervised-Learning/main_interactive.py`、`Self-Supervised-Learning/model/LSTM.py`
- 这一周在做什么：
  - 引入 `main_interactive.py`，开始尝试交互式训练流程。
  - 新增或修复 `LSTM.py`，说明模型路线从单纯 CNN 扩展到了时序模型。
- 阶段目标：探索更适合 ECG 序列的训练方式和模型结构，为后续 CNN、LSTM、Transformer 对比做准备。

### 第 9 周（2026-02-02 ~ 2026-02-08）

- 代表提交：`update`
- 主要改动目录：`Self-Supervised-Learning/main_interactive_two_version.py`
- 这一周在做什么：
  - 增加双版本交互式训练脚本，说明你开始比较不同交互策略或训练版本之间的差异。
- 阶段目标：把交互式训练实验从单一实现扩展成可对照的多个版本。

### 第 10 周（2026-02-23 ~ 2026-03-01）

- 代表提交：`update`、`fix 40->20`、`fix`、`update`
- 主要改动目录：`Self-Supervised-Learning/main_interactive.py`、`main_interactive_tf.py`、`main_interactive_two_version.py`、`model/Transformer.py`、`main_tf.py`
- 这一周在做什么：
  - 正式把 Transformer 路线加进来，新增 `main_tf.py` 和 `model/Transformer.py`。
  - 并行维护交互式 CNN/LSTM/Transformer 训练脚本。
  - 提交里出现 `40->20`，说明你对关键输入长度、窗口大小或序列参数做了修正。
  - 保存 `training.log`，开始更系统地跟踪训练过程。
- 阶段目标：从 CNN/LSTM 扩展到 Transformer，并优化训练参数，让多模型方案进入同一套实验框架。

### 第 11 周（2026-03-02 ~ 2026-03-08）

- 代表提交：`fix`
- 主要改动目录：`Self-Supervised-Learning/`、`test_cnn.py`、`test_cnn_lstm.py`
- 这一周在做什么：
  - 修复交互式两版本训练、LSTM 主程序和 CNN 模型相关问题。
  - 输出多张性能对比图，例如交互迭代表现图、单 CNN 基线曲线和完整轨迹图。
  - 补充 `test_cnn.py` 和 `test_cnn_lstm.py`，说明你开始为模型对比准备单独测试脚本。
- 阶段目标：让交互式实验和单模型基线都能稳定出结果，便于定量比较哪条路线更好。

### 第 12 周（2026-03-09 ~ 2026-03-15）

- 代表提交：`fix`、`fix cnn acc`、`test`、`update CNN + LSTM -> 0.94`
- 主要改动目录：`Self-Supervised-Learning/`、`doc.py`
- 这一周在做什么：
  - 集中修复 CNN 精度问题，继续调交互式 Transformer 流程。
  - 增加测试提交，说明这一周重点是验证模型效果而不是大改结构。
  - 明确记录 `CNN + LSTM -> 0.94`，说明组合模型表现达到了一个阶段性较优结果。
  - 同时更新 `doc.py`，可能在整理实验输出或生成说明文档。
- 阶段目标：把模型效果推到较高精度，并形成可展示的阶段性结果。

### 第 13 周（2026-03-23 ~ 2026-03-29）

- 代表提交：`fix tf_model && train -> 0.92`
- 主要改动目录：`Self-Supervised-Learning/main_interactive_tf.py`、`Self-Supervised-Learning/model/Transformer.py`
- 这一周在做什么：
  - 集中修复 Transformer 模型和训练流程。
  - 提交说明直接写到 `0.92`，说明这周的工作目标很明确，就是把 Transformer 路线训练到可接受精度。
- 阶段目标：补齐 Transformer 实验结果，让它能和 CNN、CNN+LSTM 形成直接对比。

### 第 14 周（2026-04-06 ~ 2026-04-12）

- 代表提交：`Init`、`update cnn->0.85 cnn_lstm->0.86`、`update cnn&lstm->0.92`、`cnn&lstm->0.93 cnn->0.9`、`update 周报`
- 主要改动目录：`MIT-Learning/`
- 这一周在做什么：
  - 新建 `MIT-Learning/` 项目，把数据处理模块、CNN/LSTM/Transformer 模型和主程序单独整理出来。
  - 快速推进 MIT 数据集上的新一轮实验，从提交信息可以看出连续几次提升模型表现。
  - `cnn->0.85`、`cnn_lstm->0.86`、`cnn&lstm->0.92/0.93` 说明这一周主要在做模型结构调优和训练策略改进。
  - 同时更新周报，开始把研究进展文档化。
- 阶段目标：在新的 MIT-Learning 实验线中复现实验，并把 CNN+LSTM 的效果进一步做高。

### 第 15 周（2026-04-13 ~ 2026-04-19）

- 代表提交：`update md`
- 这一周在做什么：
  - 主要是文档维护，更新 Markdown 说明材料。
- 阶段目标：整理阶段成果，为汇报或论文撰写做准备。

### 第 16 周（2026-04-20 ~ 2026-04-26）

- 代表提交：`update readme`
- 这一周在做什么：
  - 更新项目 README，补充项目说明、运行方式或实验介绍。
- 阶段目标：提升项目可读性，方便展示、复现和后续交接。

### 第 17 周（2026-05-18 ~ 2026-05-24）

- 代表提交：`update`、`del`
- 主要改动目录：`MIT-Learning/`、`Self-Supervised-Learning/results/`
- 这一周在做什么：
  - 更新 `MIT-Learning/main_lstm.py`，继续维护 LSTM 路线。
  - 同时调整或删除部分 `Self-Supervised-Learning` 的结果文件，说明这周偏向结果整理和收尾。
- 阶段目标：在项目后期做代码和实验结果的清理，保留更有代表性的实现与输出。

## 三、可以直接写进周报的简版总结

如果要把这些内容进一步压缩成“周报口吻”，可以概括为：

1. 前期主要完成 ECG 数据读取、MIT-BIH/CPSC 数据接入、可视化检查和项目基础代码搭建。
2. 中期重点做传统房颤检测算法、CNN 训练流程重构、5 折交叉验证和结果输出。
3. 随后将工作重心转向 Self-Supervised-Learning，持续完成伪标签筛选、半监督训练、交互式训练、LSTM/Transformer 扩展和多模型对比。
4. 后期新建 MIT-Learning 分支，对 MIT 数据集上的 CNN、CNN+LSTM 等模型继续调优，并同步整理周报、README 和实验结果。

## 四、建议你在正式汇报时的表述方式

你可以把整个项目过程概括成下面这条主线：

“我前期先完成 ECG 数据处理、样本可视化和基础模型代码搭建；中期围绕 MIT-BIH 和 CPSC 数据集完成传统房颤检测与 CNN 监督训练实验；随后把重点转向自监督/半监督学习，扩展了交互式训练、LSTM 和 Transformer 等模型，并持续做参数修正和精度提升；后期又单独建立 MIT-Learning 分支，对 CNN 和 CNN+LSTM 结果继续优化，同时完善了 README、周报和实验结果整理。”