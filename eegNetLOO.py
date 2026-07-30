import os
import glob
import numpy as np
import pandas as pd
import traceback
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

DATA_ROOT   = r"C:\abalaji\bichat\ORIGINAL_DATA\chunks_20"
CLASSES     = ["0", "1"]
CHANS       = 16
FS          = 256
NB_CLASSES  = 2
RANDOM_SEED = 42

TARGET_LENGTH = 15360

N_PER_CLASS = 1

BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3
WEIGHT_DECAY = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_one_file(filepath, label_col_name="label"):
    df = pd.read_csv(filepath)

    if df.columns.duplicated().any():
        cols = pd.Series(df.columns)
        for dup in df.columns[df.columns.duplicated()].unique():
            dup_idx = cols[cols == dup].index.values.tolist()
            cols.loc[dup_idx] = [dup + f'_{i}' if i != 0 else dup for i in range(len(dup_idx))]
        df.columns = cols

    label_col = None
    for col in df.columns:
        if col.strip().lower() == label_col_name.lower():
            label_col = col
            break

    file_label = None
    if label_col is not None:
        label_data = df[label_col]
        if isinstance(label_data, pd.DataFrame):
            label_data = label_data.iloc[:, 0]
        file_label = label_data.iloc[0]
        df = df.drop(columns=[label_col])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    channel_names = list(df.columns)
    arr = df.values.T

    if arr.shape[0] != CHANS:
        raise ValueError(f"Expected {CHANS} channels, got {arr.shape[0]} in {filepath}")

    return arr, channel_names, file_label


def load_dataset(data_root, classes):
    X, y, patient_ids = [], [], []
    channel_names_ref = None

    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_root, class_name)
        files = sorted(glob.glob(os.path.join(class_dir, "*.csv")))

        if len(files) == 0:
            print(f"WARNING: no files found in {class_dir}")

        for f in files:
            try:
                arr, channel_names, file_label = load_one_file(f)
            except Exception as e:
                print(f"Skipping {f}: {e}")
                traceback.print_exc()
                continue

            if channel_names_ref is None:
                channel_names_ref = channel_names

            if file_label is not None and int(file_label) != label_idx:
                print(f"WARNING: label mismatch in {f} — folder says {label_idx}, "
                      f"file column says {file_label}. Using folder label.")

            X.append(arr)
            y.append(label_idx)

            fname = os.path.basename(f)
            patient_id = fname.split("_")[0]
            patient_ids.append(patient_id)

    return X, np.array(y), np.array(patient_ids), channel_names_ref


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MaxNormConstraint:
    def __init__(self, max_val, dim=0):
        self.max_val = max_val
        self.dim = dim

    def __call__(self, module):
        with torch.no_grad():
            w = module.weight
            norm = w.norm(2, dim=self.dim, keepdim=True).clamp(min=1e-8)
            desired = torch.clamp(norm, max=self.max_val)
            w *= (desired / norm)


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, depth_multiplier, kernel_size, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * depth_multiplier,
            kernel_size=kernel_size, groups=in_channels, bias=bias
        )

    def forward(self, x):
        return self.conv(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                    padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super().__init__()

        self.dropoutType = nn.Dropout2d if dropoutType == 'SpatialDropout2D' else nn.Dropout

        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength),
                                padding=(0, kernLength // 2), bias=False)
        self.bn1   = nn.BatchNorm2d(F1)

        self.depthwise = DepthwiseConv2d(F1, D, kernel_size=(Chans, 1), bias=False)
        self.bn2       = nn.BatchNorm2d(F1 * D)
        self.pool1     = nn.AvgPool2d((1, 4))
        self.drop1     = self.dropoutType(dropoutRate)

        self.separable = SeparableConv2d(F1 * D, F2, kernel_size=(1, 16),
                                          padding=(0, 8), bias=False)
        self.bn3    = nn.BatchNorm2d(F2)
        self.pool2  = nn.AvgPool2d((1, 8))
        self.drop2  = self.dropoutType(dropoutRate)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            out = self._forward_features(dummy)
            flat_dim = out.shape[1]

        self.dense = nn.Linear(flat_dim, nb_classes)
        self._depthwise_constraint = MaxNormConstraint(1.0, dim=(1, 2, 3))
        self._dense_constraint     = MaxNormConstraint(norm_rate, dim=1)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(1)

    def forward(self, x):
        x = self._forward_features(x)
        return self.dense(x)

    def apply_constraints(self):
        self._depthwise_constraint(self.depthwise.conv)
        self._dense_constraint(self.dense)


def train_one_fold(X_train, y_train, Chans, Samples):
    model = EEGNet(nb_classes=NB_CLASSES, Chans=Chans, Samples=Samples,
                   kernLength=FS // 2, F1=8, D=2, F2=16,
                   dropoutRate=0.5, dropoutType='Dropout').to(DEVICE)

    loader = DataLoader(EEGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    classes_present = np.unique(y_train)
    if len(classes_present) > 1:
        w = compute_class_weight('balanced', classes=classes_present, y=y_train)
        weight_full = np.ones(NB_CLASSES, dtype=np.float32)
        weight_full[classes_present] = w
    else:
        weight_full = np.ones(NB_CLASSES, dtype=np.float32)
    class_weights_t = torch.tensor(weight_full, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            model.apply_constraints()

            ep_loss += loss.item() * X_b.size(0)
            ep_correct += (logits.argmax(dim=1) == y_b).sum().item()
            ep_total += y_b.size(0)

        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"    epoch {epoch:03d}/{EPOCHS}  loss={ep_loss/ep_total:.4f}  acc={100*ep_correct/ep_total:.1f}%")

    return model


def evaluate_fold(model, X_test, y_test):
    model.eval()
    loader = DataLoader(EEGDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(DEVICE)
            logits = model(X_b)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds)


print("Loading data...")
X_list, y, patient_ids, channel_names = load_dataset(DATA_ROOT, CLASSES)
print(f"Channel names (in order): {channel_names}")

lengths = [arr.shape[1] for arr in X_list]
keep_mask = [length == TARGET_LENGTH for length in lengths]
n_kept = sum(keep_mask)
if n_kept == 0:
    raise RuntimeError(f"No chunks found with length == {TARGET_LENGTH}.")
print(f"Keeping only chunks with length == {TARGET_LENGTH}: {n_kept} kept, {len(lengths)-n_kept} dropped")

X_list      = [arr for arr, keep in zip(X_list, keep_mask) if keep]
y           = y[keep_mask]
patient_ids = patient_ids[keep_mask]

SAMPLES = TARGET_LENGTH
X = np.stack(X_list, axis=0).astype(np.float32)
X = X[:, np.newaxis, :, :]
print(f"Final data shape: {X.shape}, labels shape: {y.shape}")

unique_pids = np.unique(patient_ids)
pid_to_label = {}
for pid in unique_pids:
    labels_for_pid = np.unique(y[patient_ids == pid])
    pid_to_label[pid] = labels_for_pid[0]

pids_class0 = sorted([p for p in unique_pids if pid_to_label[p] == 0])
pids_class1 = sorted([p for p in unique_pids if pid_to_label[p] == 1])

rng = np.random.RandomState(RANDOM_SEED)
if len(pids_class0) < N_PER_CLASS or len(pids_class1) < N_PER_CLASS:
    raise RuntimeError(f"Not enough patients: class0={len(pids_class0)}, class1={len(pids_class1)}, need {N_PER_CLASS} each")

selected_class0 = list(rng.choice(pids_class0, size=N_PER_CLASS, replace=False))
selected_class1 = list(rng.choice(pids_class1, size=N_PER_CLASS, replace=False))
selected_pids = selected_class0 + selected_class1
print(f"\nSelected patients (class 0): {selected_class0}")
print(f"Selected patients (class 1): {selected_class1}")

fold_results = []
overall_true, overall_pred_chunks = [], []

for held_out_pid in selected_pids:
    print(f"\n=== Leave-one-out fold: held-out patient {held_out_pid} ===")

    train_mask = patient_ids != held_out_pid
    test_mask  = patient_ids == held_out_pid

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    true_label = y_test[0]
    print(f"  train chunks: {X_train.shape[0]}  (class dist {np.bincount(y_train, minlength=2)})")
    print(f"  test chunks:  {X_test.shape[0]}  true patient label: {true_label}")

    model = train_one_fold(X_train, y_train, Chans=CHANS, Samples=SAMPLES)
    chunk_preds = evaluate_fold(model, X_test, y_test)

    majority_pred = np.bincount(chunk_preds, minlength=NB_CLASSES).argmax()
    n_correct_chunks = (chunk_preds == true_label).sum()

    print(f"  chunk preds: {chunk_preds.tolist()}")
    print(f"  chunk-level acc: {n_correct_chunks}/{len(chunk_preds)}")
    print(f"  majority-vote prediction: {majority_pred}  (true: {true_label})  "
          f"{'CORRECT' if majority_pred == true_label else 'WRONG'}")

    fold_results.append({
        "patient": held_out_pid,
        "true": true_label,
        "pred": majority_pred,
        "n_chunks": len(chunk_preds),
        "n_correct_chunks": int(n_correct_chunks),
    })

    overall_true.append(true_label)
    overall_pred_chunks.extend(chunk_preds.tolist())

print("\n\n===================== SUMMARY =====================")
patient_true = np.array([r["true"] for r in fold_results])
patient_pred = np.array([r["pred"] for r in fold_results])

for r in fold_results:
    status = "CORRECT" if r["true"] == r["pred"] else "WRONG"
    print(f"  Patient {r['patient']}: true={r['true']} pred={r['pred']} "
          f"({r['n_correct_chunks']}/{r['n_chunks']} chunks correct) -> {status}")

acc = (patient_true == patient_pred).mean()
print(f"\nLOPO patient-level accuracy: {acc:.4f} ({(patient_true == patient_pred).sum()}/{len(patient_true)})")
print("\nClassification report (patient-level, LOPO):")
print(classification_report(patient_true, patient_pred, target_names=CLASSES, zero_division=0))
print("Confusion matrix (patient-level, LOPO):")
print(confusion_matrix(patient_true, patient_pred))