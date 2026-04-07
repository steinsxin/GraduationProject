import matplotlib.pyplot as plt
import numpy as np

# Set style for academic publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.2

# 1. Data Preparation
iterations = np.arange(1, 11) # Rounds 1 to 10

cnn_acc = [0.703, 0.836, 0.927, 0.908, 0.936, 0.932, 0.950, 0.944, 0.942, 0.940]
lstm_acc = [0.705, 0.825, 0.900, 0.932, 0.929, 0.947, 0.941, 0.946, 0.950, 0.939]
ensemble_acc = [0.704, 0.841, 0.918, 0.926, 0.935, 0.944, 0.951, 0.952, 0.951, 0.949]

# 2. Create the Plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot CNN
ax.plot(iterations, cnn_acc, marker='o', linestyle='-', color='#1f77b4', 
        linewidth=2, markersize=6, label='CNN (Morphology)', alpha=0.8)

# Plot LSTM
ax.plot(iterations, lstm_acc, marker='s', linestyle='--', color='#d62728', 
        linewidth=2, markersize=6, label='LSTM (Temporal)', alpha=0.8)

# Plot Ensemble (Interaction) - Highlighted
ax.plot(iterations, ensemble_acc, marker='^', linestyle='-', color='#2ca02c', 
        linewidth=3, markersize=8, label='CNN-LSTM Interactive (Ensemble)', zorder=5)

# 3. Annotations and Visual Aids

# Highlight the "Stability & Superiority" region (Rounds 6-10)
ax.axvspan(6, 10, color='green', alpha=0.08, label='Stable High-Performance Region')

# Annotate the Peak Performance
max_ens_val = max(ensemble_acc)
max_ens_idx = ensemble_acc.index(max_ens_val) + 1 # Convert to 1-based index
ax.annotate(f'Peak Acc: {max_ens_val:.3f}\n(Round {max_ens_idx})', 
            xy=(max_ens_idx, max_ens_val), 
            xytext=(max_ens_idx + 1, max_ens_val - 0.04),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10, fontweight='bold', color='darkgreen',
            bbox=dict(facecolor='white', edgecolor='green', alpha=0.8, boxstyle='round,pad=0.5'))

# Annotate the "Cross-Feeding" Effect (e.g., Round 3 where CNN spikes but Ensemble balances)
# Or simply highlight that Ensemble avoids the dips seen in single models
# Example: Round 4 CNN dip vs Ensemble stability
ax.text(4.2, 0.915, 'Interaction Effect:\nEnsemble stabilizes\nindividual fluctuations', 
        fontsize=9, color='darkgray', style='italic',
        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.5'))

# 4. Formatting
ax.set_title('CNN-LSTM Interactive Iteration: Performance Evolution', fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel('Iteration Round', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')

ax.set_xticks(iterations)
ax.set_ylim(0.65, 0.97)  # Set Y-axis to clearly show differences
ax.legend(loc='lower right', fontsize=10, frameon=True, shadow=False)

# Grid settings
ax.grid(True, which='major', linestyle='--', alpha=0.6)
ax.grid(True, which='minor', linestyle=':', alpha=0.3)

# Save and Show
output_filename = 'interactive_iteration_performance.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot successfully saved as '{output_filename}'")

plt.show()