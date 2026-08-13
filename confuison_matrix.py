import matplotlib.pyplot as plt
import numpy as np

cm = np.array([[3, 2],
               [2, 3]])
total_patients = 10
percentages = (cm / total_patients) * 100
labels = np.array([
    [f"{cm[0,0]}\n({percentages[0,0]:.1f}%)", f"{cm[0,1]}\n({percentages[0,1]:.1f}%)"],
    [f"{cm[1,0]}\n({percentages[1,0]:.1f}%)", f"{cm[1,1]}\n({percentages[1,1]:.1f}%)"]
])

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=5)
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75)
cbar.ax.set_ylabel('Number of Patients', rotation=-90, va="bottom", fontsize=11)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['0 (Alive)', '1 (Dead)'], fontsize=11)
ax.set_yticklabels(['0 (Alive)', '1 (Dead)'], fontsize=11)

fig.suptitle('Patient-level Confusion Matrix (Majority Vote Across Chunks)',
             fontsize=14, fontweight='bold', y=0.97)
ax.set_title(f'Total Patients: {total_patients}', fontsize=12, pad=35)
ax.set_xlabel('Predicted Label (Majority Vote)', fontsize=12, labelpad=15)
ax.xaxis.set_label_position('top')
ax.set_ylabel('True Label', fontsize=12, labelpad=15)

for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > 7 else "black"
        ax.text(j, i, labels[i, j], ha="center", va="center",
                 color=color, fontsize=14, fontweight='bold')

ax.text(0.5, -0.12, "Rows: True Label   |   Columns: Predicted Label",
         ha="center", transform=ax.transAxes, fontsize=10)

metrics_text = (
    "Accuracy: 89.5% (17/19)               Recall (Alive, 0): 93.3% (14/15)\n"
    "Balanced Accuracy: 84.2%         Recall (Dead, 1): 75.0% (3/4)\n"
    "Macro Avg Recall: 84.2%            Precision (Alive, 0): 93.3% (14/15)\n"
    "Total Patients: 19                        Precision (Dead, 1): 75.0% (3/4)"
)
fig.text(0.5, 0.06, metrics_text, ha="center", fontsize=10,
          bbox=dict(boxstyle="round,pad=0.8", facecolor="none", edgecolor="gray"))
fig.text(0.5, 0.015, "Labels: 0 = Alive, 1 = Dead", ha="center", fontsize=10)

fig.subplots_adjust(top=0.82, bottom=0.28)
fig.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()