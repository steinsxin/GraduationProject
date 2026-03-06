import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'  # 确保英文字符显示正常
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300

# 1. 数据准备
rounds = np.arange(1, 14) # 1 到 13
accuracy = np.array([
    0.7140, 0.8610, 0.9030, 0.9120, # R1-R4: Rapid Rise
    0.9110, 0.9260, 0.9150, 0.9110, 0.9120, 0.9060, # R5-R10: Saturation/Fluctuation
    0.8110, 0.7460, 0.7560 # R11-R13: Collapse
])

f1_score = np.array([
    0.7776, 0.8764, 0.9087, 0.9182, # R1-R4
    0.9156, 0.9306, 0.9188, 0.9125, 0.9139, 0.9089, # R5-R10
    0.8355, 0.7918, 0.7993 # R11-R13
])

# 2. 创建图表
fig, ax = plt.subplots()

# 绘制 Accuracy
ax.plot(rounds, accuracy, marker='o', linestyle='-', color='#1f77b4', 
        linewidth=2.5, markersize=8, label='Accuracy', zorder=3)

# 绘制 F1-Score
ax.plot(rounds, f1_score, marker='s', linestyle='--', color='#d62728', 
        linewidth=2.5, markersize=8, label='F1-Score', zorder=3)

# 3. 添加关键区域标注

# A. 标注“性能饱和区” (Round 5-10)
ax.axvspan(4.5, 10.5, color='orange', alpha=0.15, label='Saturation Phase (Plateau)')
ax.text(7.5, 0.94, 'Saturation Phase:\nPerformance stalls\naround 0.91-0.92', 
        fontsize=10, color='darkorange', weight='bold', style='italic',
        ha='center', va='top',
        bbox=dict(facecolor='white', edgecolor='orange', alpha=0.8, boxstyle='round,pad=0.5'))

# B. 标注“性能崩溃区” (Round 11-13) - 这是单模型半监督的典型失败模式
ax.axvspan(10.5, 13.5, color='red', alpha=0.15, label='Degradation Phase (Collapse)')
ax.text(12, 0.78, 'Performance Collapse:\nError accumulation\nleads to sharp drop', 
        fontsize=10, color='darkred', weight='bold', style='italic',
        ha='center', va='bottom',
        bbox=dict(facecolor='white', edgecolor='red', alpha=0.8, boxstyle='round,pad=0.5'))

# C. 标注峰值点 (Round 6)
peak_idx = 5 # Index for Round 6 (0-based)
ax.annotate(f'Peak Acc: {accuracy[peak_idx]:.3f}\n(Round 6)', 
            xy=(rounds[peak_idx], accuracy[peak_idx]), 
            xytext=(rounds[peak_idx] + 1.5, accuracy[peak_idx] + 0.03),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10, fontweight='bold', color='darkgreen')

# D. 标注起始点和崩溃点
ax.scatter([1, 13], [accuracy[0], accuracy[-1]], color='black', s=80, zorder=4, edgecolors='white', linewidth=1.5)
ax.text(1, accuracy[0] - 0.04, f'Start: {accuracy[0]:.3f}', ha='center', fontsize=9)
ax.text(13, accuracy[-1] + 0.02, f'End: {accuracy[-1]:.3f}', ha='center', fontsize=9, color='darkred', weight='bold')

# 4. 格式化
ax.set_title('Single CNN Baseline: Full Training Trajectory (13 Rounds)', fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel('Training Iteration Round', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance Metric', fontsize=12, fontweight='bold')

ax.set_xticks(rounds)
ax.set_ylim(0.65, 0.96) # 调整Y轴以清晰展示崩溃过程
ax.legend(loc='upper right', fontsize=10, frameon=True, shadow=False)

# 网格设置
ax.grid(True, which='major', linestyle='--', alpha=0.6)
ax.grid(True, which='minor', linestyle=':', alpha=0.3)

# 保存与显示
output_filename = 'single_cnn_baseline_full_trajectory.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved as '{output_filename}'")

plt.show()