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

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

DATA_ROOT = r"D:\abalaji\chunks_20"
CLINICAL_PATH = r"D:\abalaji\EEG_bichat_CNN\clinical_data_labeled_imputed.xlsx"
CLINICAL_FEATURE_COLS = ["SEXE", "AGE", "HTA", "Diabète", "Obésité",
                          "Valeur NSE 1 (µg/L)", "Valeur NSE 2 (µg/L)"]

CLASSES = ["0", "1"]
CHANS = 16
FS = 256
NB_CLASSES = 2
RANDOM_SEED = 42

TARGET_LENGTH = 15360

TEST_SIZE = 0.2
VAL_SIZE = 0.2

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 0.0
PATIENCE = 15
USE_EARLY_STOPPING = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_one_file(filepath, label_col_name="label"):
    df = pd.read_csv(filepath)

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        print(f"WARNING: duplicate columns in {filepath}: {dupes}")
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

    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Non-numeric/NaN values found in {filepath} after coercion, columns: {bad_cols}")

    channel_names = list(df.columns)
    arr = df.values.T

    if arr.shape[0] != CHANS:
        raise ValueError(f"Expected {CHANS} channels, got {arr.shape[0]} in {filepath} "
                          f"(columns found: {channel_names})")

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
            elif channel_names != channel_names_ref:
                print(f"WARNING: channel order/names differ in {f}")

            if file_label is not None:
                try:
                    if int(file_label) != label_idx:
                        print(f"WARNING: label mismatch in {f} — folder says {label_idx}, "
                              f"file column says {file_label}. Using folder label.")
                except (ValueError, TypeError):
                    print(f"WARNING: unreadable label column value in {f}: {file_label!r}")

            X.append(arr)
            y.append(label_idx)

            fname = os.path.basename(f)
            patient_id = fname.split("_")[0]
            patient_ids.append(patient_id)

    return X, np.array(y), np.array(patient_ids), channel_names_ref


print("Loading EEG data...")
X_list, y, patient_ids, channel_names = load_dataset(DATA_ROOT, CLASSES)
print(f"Channel names (in order): {channel_names}")

lengths = [arr.shape[1] for arr in X_list]
print(f"Chunk lengths -> min: {min(lengths)}, max: {max(lengths)}, unique: {len(set(lengths))}")
print(f"Length distribution: {Counter(lengths)}")

keep_mask = [length == TARGET_LENGTH for length in lengths]
n_kept = sum(keep_mask)
n_dropped = len(lengths) - n_kept
print(f"Keeping only chunks with length == {TARGET_LENGTH}: {n_kept} kept, {n_dropped} dropped")

if n_kept == 0:
    raise RuntimeError(f"No chunks found with length == {TARGET_LENGTH}. Check TARGET_LENGTH.")

X_list = [arr for arr, keep in zip(X_list, keep_mask) if keep]
y = y[keep_mask]
patient_ids = patient_ids[keep_mask]

SAMPLES = TARGET_LENGTH
X = np.stack(X_list, axis=0).astype(np.float32)
X = X[:, np.newaxis, :, :]

print(f"Final EEG data shape: {X.shape}, labels shape: {y.shape}")
print(f"Class distribution (chunks): {np.bincount(y)}")
print(f"Number of unique patients: {len(np.unique(patient_ids))}")

print("\nLoading clinical data...")
clinical_df = pd.read_excel(CLINICAL_PATH)
clinical_df["N_PAT"] = clinical_df["N_PAT"].astype(str)
clinical_lookup = clinical_df.set_index("N_PAT")[CLINICAL_FEATURE_COLS]

missing_clinical = sorted(set(pid for pid in np.unique(patient_ids) if pid not in clinical_lookup.index))
if missing_clinical:
    raise ValueError(f"No clinical row found for patients: {missing_clinical}")

clinical_raw = np.stack([clinical_lookup.loc[pid].values for pid in patient_ids]).astype(np.float32)
print(f"Clinical feature matrix shape: {clinical_raw.shape}")

unique_pids = np.unique(patient_ids)
pid_to_label = {}
for pid in unique_pids:
    mask = patient_ids == pid
    labels_for_pid = np.unique(y[mask])
    if len(labels_for_pid) > 1:
        print(f"WARNING: patient {pid} has multiple labels {labels_for_pid}")
    pid_to_label[pid] = labels_for_pid[0]

print(f"\nPatient-level class distribution: {np.bincount([pid_to_label[p] for p in unique_pids])}")

gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
train_pool_idx, test_idx = next(gss_test.split(X, y, groups=patient_ids))

train_pool_pids = np.unique(patient_ids[train_pool_idx])
test_pids = np.unique(patient_ids[test_idx])
assert set(train_pool_pids).isdisjoint(set(test_pids)), "Leakage: patient in both train and test!"

X_pool, y_pool, pid_pool = X[train_pool_idx], y[train_pool_idx], patient_ids[train_pool_idx]
Xclin_pool = clinical_raw[train_pool_idx]

gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_SEED)
train_idx_rel, val_idx_rel = next(gss_val.split(X_pool, y_pool, groups=pid_pool))

X_train, y_train = X_pool[train_idx_rel], y_pool[train_idx_rel]
X_val, y_val = X_pool[val_idx_rel], y_pool[val_idx_rel]
X_test, y_test = X[test_idx], y[test_idx]

Xclin_train_raw = Xclin_pool[train_idx_rel]
Xclin_val_raw = Xclin_pool[val_idx_rel]
Xclin_test_raw = clinical_raw[test_idx]

train_pids_final = np.unique(pid_pool[train_idx_rel])
val_pids_final = np.unique(pid_pool[val_idx_rel])

assert set(train_pids_final).isdisjoint(set(val_pids_final))
assert set(train_pids_final).isdisjoint(set(test_pids))
assert set(val_pids_final).isdisjoint(set(test_pids))

clin_scaler = StandardScaler()
clin_scaler.fit(clinical_lookup.loc[train_pids_final].values.astype(np.float32))
Xclin_train = clin_scaler.transform(Xclin_train_raw).astype(np.float32)
Xclin_val = clin_scaler.transform(Xclin_val_raw).astype(np.float32)
Xclin_test = clin_scaler.transform(Xclin_test_raw).astype(np.float32)

eeg_mean = X_train.mean(axis=(0, 1, 3), keepdims=True)
eeg_std = X_train.std(axis=(0, 1, 3), keepdims=True)
eeg_std[eeg_std < 1e-8] = 1e-8
X_train = (X_train - eeg_mean) / eeg_std
X_val = (X_val - eeg_mean) / eeg_std
X_test = (X_test - eeg_mean) / eeg_std

print(f"\nPatients  -> train: {len(train_pids_final)}, val: {len(val_pids_final)}, test: {len(test_pids)}")
print(f"Chunks    -> train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}")
print(f"Class dist -> train: {np.bincount(y_train)}, val: {np.bincount(y_val)}, test: {np.bincount(y_test)}")


class EEGClinicalDataset(Dataset):
    def __init__(self, X, X_clin, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X_clin = torch.tensor(X_clin, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.X_clin[idx], self.y[idx]


train_loader = DataLoader(EEGClinicalDataset(X_train, Xclin_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(EEGClinicalDataset(X_val, Xclin_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(EEGClinicalDataset(X_test, Xclin_test, y_test), batch_size=BATCH_SIZE, shuffle=False)


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


def same_pad_1d(x, kernel_size):
    total_pad = kernel_size - 1
    left = total_pad // 2
    right = total_pad - left
    return F.pad(x, (left, right))


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
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                    padding=0, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = same_pad_1d(x, self.kernel_size[1])
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGNetFeatureExtractor(nn.Module):
    def __init__(self, Chans=16, Samples=15360, dropoutRate=0.5, kernLength=64,
                 F1=8, D=2, F2=16, dropoutType='Dropout'):
        super().__init__()

        if dropoutType == 'SpatialDropout2D':
            self.dropoutType = nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.dropoutType = nn.Dropout
        else:
            raise ValueError("dropoutType must be 'SpatialDropout2D' or 'Dropout'")

        self.kernLength = kernLength
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = DepthwiseConv2d(F1, D, kernel_size=(Chans, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = self.dropoutType(dropoutRate)

        self.separable = SeparableConv2d(F1 * D, F2, kernel_size=(1, 16))
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = self.dropoutType(dropoutRate)

        self._depthwise_constraint = MaxNormConstraint(1.0, dim=(1, 2, 3))

        with torch.no_grad():
            was_training = self.training
            self.eval()
            dummy = torch.zeros(1, 1, Chans, Samples)
            out = self.forward(dummy)
            self.out_dim = out.shape[1]
            self.train(was_training)

    def forward(self, x):
        x = same_pad_1d(x, self.kernLength)
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

    def apply_constraints(self):
        self._depthwise_constraint(self.depthwise.conv)


class ClinicalMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=16, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiModalEEGNet(nn.Module):
    def __init__(self, nb_classes, n_clinical_features, Chans=16, Samples=15360,
                 kernLength=128, F1=8, D=2, F2=16, dropoutRate=0.5,
                 dropoutType='Dropout', clinical_hidden=32, clinical_out=16,
                 fusion_hidden=32, norm_rate=0.25):
        super().__init__()
        self.eeg_branch = EEGNetFeatureExtractor(
            Chans=Chans, Samples=Samples, dropoutRate=dropoutRate,
            kernLength=kernLength, F1=F1, D=D, F2=F2, dropoutType=dropoutType
        )
        self.clinical_branch = ClinicalMLP(n_clinical_features, clinical_hidden, clinical_out)

        fusion_in = self.eeg_branch.out_dim + clinical_out
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropoutRate),
            nn.Linear(fusion_hidden, nb_classes)
        )
        self._dense_constraint = MaxNormConstraint(norm_rate, dim=1)

    def forward(self, x_eeg, x_clin):
        eeg_feat = self.eeg_branch(x_eeg)
        clin_feat = self.clinical_branch(x_clin)
        fused = torch.cat([eeg_feat, clin_feat], dim=1)
        return self.classifier(fused)

    def apply_constraints(self):
        self.eeg_branch.apply_constraints()
        self._dense_constraint(self.classifier[-1])


model = MultiModalEEGNet(
    nb_classes=NB_CLASSES,
    n_clinical_features=len(CLINICAL_FEATURE_COLS),
    Chans=CHANS,
    Samples=SAMPLES,
    kernLength=FS // 2,
    F1=8, D=2, F2=16,
    dropoutRate=0.5,
    dropoutType='Dropout'
).to(DEVICE)

print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params:,}")

class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: {class_weights}")

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_val_loss = float('inf')
best_state = None
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for X_b, Xclin_b, y_b in train_loader:
        X_b, Xclin_b, y_b = X_b.to(DEVICE), Xclin_b.to(DEVICE), y_b.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_b, Xclin_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        model.apply_constraints()

        tr_loss += loss.item() * X_b.size(0)
        preds = logits.argmax(dim=1)
        tr_correct += (preds == y_b).sum().item()
        tr_total += y_b.size(0)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for X_b, Xclin_b, y_b in val_loader:
            X_b, Xclin_b, y_b = X_b.to(DEVICE), Xclin_b.to(DEVICE), y_b.to(DEVICE)
            logits = model(X_b, Xclin_b)
            loss = criterion(logits, y_b)
            val_loss += loss.item() * X_b.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == y_b).sum().item()
            val_total += y_b.size(0)

    avg_tr_loss = tr_loss / tr_total
    avg_val_loss = val_loss / val_total
    tr_acc = 100.0 * tr_correct / tr_total
    val_acc = 100.0 * val_correct / val_total

    print(f"Epoch [{epoch:03d}/{EPOCHS}]  "
          f"Train Loss: {avg_tr_loss:.4f} Acc: {tr_acc:.1f}%  |  "
          f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}%")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss - 1e-4:
        best_val_loss = avg_val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if USE_EARLY_STOPPING and no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    torch.save(best_state, "best_multimodal_eegnet_model.pt")
print(f"Best val loss: {best_val_loss:.4f}")

model.eval()
all_logits, all_true = [], []
with torch.no_grad():
    for X_b, Xclin_b, y_b in test_loader:
        X_b, Xclin_b = X_b.to(DEVICE), Xclin_b.to(DEVICE)
        logits = model(X_b, Xclin_b)
        all_logits.append(logits.cpu().numpy())
        all_true.append(y_b.numpy())

all_logits = np.concatenate(all_logits, axis=0)
y_test_true = np.concatenate(all_true, axis=0)
y_pred_probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
y_pred = y_pred_probs.argmax(axis=1)

test_acc = (y_pred == y_test_true).mean()
print(f"\nTest accuracy: {test_acc:.4f}")

print("\nClassification Report (chunk-level, held-out patients):")
print(classification_report(y_test_true, y_pred, target_names=CLASSES, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_true, y_pred))

if len(np.unique(y_test_true)) > 1:
    auc = roc_auc_score(y_test_true, y_pred_probs[:, 1])
    print(f"ROC-AUC: {auc:.4f}")

test_pids_per_chunk = patient_ids[test_idx]

print("\n── Patient-level results (mean predicted probability across chunks) ──")
patient_true, patient_pred = [], []
total_correct_chunks = 0
total_chunks = 0

for pid in np.unique(test_pids_per_chunk):
    mask = test_pids_per_chunk == pid
    true_label = y_test[mask][0]
    mean_probs = y_pred_probs[mask].mean(axis=0)
    pred_label = int(mean_probs.argmax())
    patient_true.append(true_label)
    patient_pred.append(pred_label)

    n_chunks = mask.sum()
    n_correct = (y_pred[mask] == y_test[mask]).sum()
    total_correct_chunks += n_correct
    total_chunks += n_chunks

    print(f"  Patient {pid}: true={true_label}, predicted(mean-prob)={pred_label}, "
          f"n_chunks={n_chunks}, chunk_preds={y_pred[mask].tolist()}, "
          f"correct_chunks={n_correct}/{n_chunks}")

print(f"\nOverall chunk-level accuracy within test patients: "
      f"{total_correct_chunks}/{total_chunks} "
      f"({100.0*total_correct_chunks/total_chunks:.1f}%)")

patient_true = np.array(patient_true)
patient_pred = np.array(patient_pred)
print(f"\nPatient-level accuracy: {(patient_true == patient_pred).mean():.4f}")
print(classification_report(patient_true, patient_pred, target_names=CLASSES, zero_division=0))