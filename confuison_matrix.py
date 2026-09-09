import matplotlib.pyplot as plt
import numpy as np

cm = np.array([[107, 15],
               [15, 62]])

row_totals = cm.sum(axis=1, keepdims=True)
total_patients = cm.sum()
percentages = (cm / row_totals) * 100

TN, FP = cm[0, 0], cm[0, 1]
FN, TP = cm[1, 0], cm[1, 1]

accuracy = (TN + TP) / total_patients
recall_0 = TN / (TN + FP)
recall_1 = TP / (TP + FN)
precision_0 = TN / (TN + FN)
precision_1 = TP / (TP + FP)
f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)
f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
balanced_accuracy = (recall_0 + recall_1) / 2
macro_precision = (precision_0 + precision_1) / 2
macro_recall = (recall_0 + recall_1) / 2
macro_f1 = (f1_0 + f1_1) / 2

labels = np.array([
    [f"{cm[0,0]}\n({percentages[0,0]:.1f}%)", f"{cm[0,1]}\n({percentages[0,1]:.1f}%)"],
    [f"{cm[1,0]}\n({percentages[1,0]:.1f}%)", f"{cm[1,1]}\n({percentages[1,1]:.1f}%)"]
])

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75)
cbar.ax.set_ylabel('Number of Patients', rotation=-90, va="bottom", fontsize=11)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['0 (Alive)', '1 (Dead)'], fontsize=11)
ax.set_yticklabels(['0 (Alive)', '1 (Dead)'], fontsize=11)

fig.suptitle('Patient-level Confusion Matrix (Indirect organoid data with RF)',
             fontsize=14, fontweight='bold', y=0.97)
ax.set_title(f'Total Patients: {total_patients}', fontsize=12, pad=35)
ax.set_xlabel('Predicted Label (Majority Vote)', fontsize=12, labelpad=15)
ax.xaxis.set_label_position('top')
ax.set_ylabel('True Label', fontsize=12, labelpad=15)

for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
        ax.text(j, i, labels[i, j], ha="center", va="center",
                 color=color, fontsize=14, fontweight='bold')

ax.text(0.5, -0.12, "Rows: True Label   |   Columns: Predicted Label",
         ha="center", transform=ax.transAxes, fontsize=10)

metrics_text = (
    f"Accuracy: {accuracy*100:.1f}% ({TN+TP}/{total_patients})               Recall (Alive, 0): {recall_0*100:.1f}% ({TN}/{TN+FP})\n"
    f"Balanced Accuracy: {balanced_accuracy*100:.1f}%         Recall (Dead, 1): {recall_1*100:.1f}% ({TP}/{TP+FN})\n"
    f"Macro Avg Recall: {macro_recall*100:.1f}%            Precision (Alive, 0): {precision_0*100:.1f}% ({TN}/{TN+FN})\n"
    f"Total Patients: {total_patients}                        Precision (Dead, 1): {precision_1*100:.1f}% ({TP}/{TP+FP})\n"
    f"Macro Avg Precision: {macro_precision*100:.1f}%       F1 (Alive, 0): {f1_0*100:.1f}%\n"
    f"Macro Avg F1: {macro_f1*100:.1f}%                        F1 (Dead, 1): {f1_1*100:.1f}%"
)
fig.text(0.5, 0.06, metrics_text, ha="center", fontsize=10,
          bbox=dict(boxstyle="round,pad=0.8", facecolor="none", edgecolor="gray"))
fig.text(0.5, 0.015, "Labels: 0 = Alive, 1 = Dead", ha="center", fontsize=10)

fig.subplots_adjust(top=0.82, bottom=0.28)
fig.savefig('img/confusion_matrix_rf_indirectOrganoid.png', bbox_inches='tight')
plt.show()