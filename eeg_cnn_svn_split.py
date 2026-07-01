"""
EEG Death vs No-Death Classification — v3 (window-wise split, multi-classifier)
CNN → Flatten/FC(128) features → {SVM, RandomForest, 1-layer NN} × {with PCA, without PCA}

Changes vs v2:
  - Patient-level split.
  - 6 classifier variants:
        SVM        (raw features)
        SVM        (PCA features)
        RandomForest (raw features)
        RandomForest (PCA features)
        1-layer NN (raw features)
        1-layer NN (PCA features)
  - Metrics for all 6 on the held-out window test set.

Folder structure:
  chunks_20/0/  ← survived  (1214_chunk_000.csv ...)
  chunks_20/1/  ← died      (2045_chunk_000.csv ...)
Each CSV: (15361, 17) — header row + 16 EEG cols + 1 extra col (dropped)
"""

import re
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score
)

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
DATA_ROOT      = Path(r"D:\abalaji\chunks_20")
N_CHANNELS     = 16
WINDOW_SIZE    = 256
STEP_SIZE      = 128
BATCH_SIZE     = 64
EPOCHS         = 50
LR            = 5e-4
WEIGHT_DECAY   = 1e-4
PCA_COMPONENTS = 32
PATIENCE       = 7
TEST_SIZE      = 0.2
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_MODEL     = "eeg_cnn_v3.pth"
SAVE_PCA       = "eeg_pca_v3.pkl"
SAVE_DIR       = Path("models_v3")
SAVE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────
def parse_filename(filename):
    match = re.match(r"(\d+)_chunk_(\d+)\.csv", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def extract_windows(chunk, label, window_size, step_size):
    windows, labels = [], []
    for start in range(0, len(chunk) - window_size, step_size):
        windows.append(chunk[start:start + window_size])
        labels.append(label)
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_all_data(data_root, window_size, step_size):
    patient_files = defaultdict(list)

    for label in [0, 1]:
        folder = data_root / str(label)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        csv_files = sorted(folder.glob("*.csv"))
        print(f"Folder /{label}: {len(csv_files)} CSV files found")
        for csv_path in csv_files:
            pid, cid = parse_filename(csv_path.name)
            if pid is None:
                print(f"  [SKIP] Unrecognized: {csv_path.name}")
                continue
            patient_files[pid].append((cid, csv_path, label))

    all_pids = sorted(patient_files.keys())
    print(f"\nTotal unique patients : {len(all_pids)}")
    pid_labels = {pid: chunks[0][2] for pid, chunks in patient_files.items()}
    lc = Counter(pid_labels.values())
    print(f"  Survived (0) : {lc[0]} patients")
    print(f"  Died     (1) : {lc[1]} patients")

    pid_to_idx = {pid: idx for idx, pid in enumerate(all_pids)}
    all_windows, all_labels, all_groups = [], [], []
    skipped = 0

    for pid in all_pids:
        pid_idx = pid_to_idx[pid]
        chunks  = sorted(patient_files[pid], key=lambda x: x[0])

        for cid, csv_path, label in chunks:
            df = pd.read_csv(csv_path, header=0)
            chunk = df.values.astype(np.float32)

            if chunk.shape[1] == N_CHANNELS + 1:
                chunk = chunk[:, :N_CHANNELS]

            if chunk.shape[1] != N_CHANNELS or chunk.shape[0] < window_size:
                print(f"  [SKIP] {csv_path.name} shape: {chunk.shape}")
                skipped += 1
                continue

            scaler = StandardScaler()
            chunk  = scaler.fit_transform(chunk)

            windows, wlabels = extract_windows(
                chunk, label, window_size, step_size
            )
            all_windows.append(windows)
            all_labels.append(wlabels)
            all_groups.extend([pid_idx] * len(windows))

        if (pid_idx + 1) % 20 == 0:
            print(f"  Processed {pid_idx+1}/{len(all_pids)} patients...")

    print(f"Skipped files : {skipped}")

    X      = np.concatenate(all_windows, axis=0)
    y      = np.concatenate(all_labels,  axis=0)
    groups = np.array(all_groups)
    X      = X[..., np.newaxis]                    # (N, 256, 16, 1)

    print(f"\n── Dataset Summary ──────────────────")
    print(f"X shape            : {X.shape}")
    print(f"Windows (died=1)   : {(y==1).sum():,}")
    print(f"Windows (alive=0)  : {(y==0).sum():,}")
    print(f"Class ratio 1/0    : {(y==1).sum()/(y==0).sum():.3f}")
    print(f"─────────────────────────────────────\n")

    return X, y, groups, pid_to_idx, pid_labels


# ─────────────────────────────────────────────────────────
# 2. PYTORCH DATASET
# ─────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, X, y):
        X_t = np.transpose(X, (0, 3, 1, 2))         # (N, 1, 256, 16)
        self.X = torch.tensor(X_t, dtype=torch.float32)
        self.y = torch.tensor(y,   dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────
# 3. CNN FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)


class EEG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  32),   # → (32, 128, 8)
            ConvBlock(32, 64),   # → (64,  64, 4)
            ConvBlock(64, 64),   # → (64,  32, 2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.head = nn.Linear(128, 1)

    def forward_features(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x                        # (batch, 128)

    def forward(self, x):
        return self.head(self.forward_features(x))


# ─────────────────────────────────────────────────────────
# 4. CNN TRAINING (with internal val split, early stopping)
# ─────────────────────────────────────────────────────────
def train_cnn(model, train_loader, val_loader, device,
              epochs, lr, weight_decay, pos_weight, patience, min_delta=1e-3):

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    pw        = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    print("── Phase 1: Training CNN ─────────────")
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item()
            preds       = (torch.sigmoid(logits) > 0.5).float()
            tr_correct += (preds == y_b).sum().item()
            tr_total   += y_b.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device)
                y_b = y_b.to(device).unsqueeze(1)
                logits      = model(X_b)
                loss        = criterion(logits, y_b)
                val_loss   += loss.item()
                preds       = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y_b).sum().item()
                val_total   += y_b.size(0)

        avg_tr_loss  = tr_loss  / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        tr_acc       = 100.0 * tr_correct  / tr_total
        val_acc      = 100.0 * val_correct / val_total

        print(f"  Epoch [{epoch:02d}/{epochs}]  "
              f"Train Loss: {avg_tr_loss:.4f} Acc: {tr_acc:.1f}%  |  "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}%")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            print(f"    (no meaningful improvement: {no_improve}/{patience})")
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement > {min_delta} for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.4f}\n")


# ─────────────────────────────────────────────────────────
# 5. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            f = model.forward_features(X_b.to(device))
            feats.append(f.cpu().numpy())
            labels.append(y_b.numpy())
    return np.concatenate(feats), np.concatenate(labels)


# ─────────────────────────────────────────────────────────
# 6. ONE-LAYER NN CLASSIFIER (single linear layer + sigmoid)
# ─────────────────────────────────────────────────────────
class OneLayerNN(nn.Module):
    """A single linear layer mapping features -> 1 logit (i.e. logistic regression
    expressed as a 1-layer neural net), trained with BCEWithLogitsLoss."""
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x)


def train_one_layer_nn(X_tr, y_tr, X_te, y_te, device,
                        epochs=100, lr=1e-3, weight_decay=1e-3, batch_size=64):
    in_dim = X_tr.shape[1]
    net = OneLayerNN(in_dim).to(device)

    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)
    pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    net.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = net(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    net.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
        logits = net(X_te_t)
        proba  = torch.sigmoid(logits).cpu().numpy().ravel()
    y_pred = (proba > 0.5).astype(int)
    return net, y_pred, proba


# ─────────────────────────────────────────────────────────
# 7. EVALUATION HELPER
# ─────────────────────────────────────────────────────────
def evaluate(name, y_te, y_pred, y_proba):
    print(f"\n── Results: {name} ────────────────────────")
    print(classification_report(
        y_te, y_pred,
        target_names=['Survived (0)', 'Died (1)']
    ))
    cm = confusion_matrix(y_te, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                 Pred:Survived  Pred:Died")
    print(f"  Actual Survived:    {cm[0,0]:5d}        {cm[0,1]:5d}")
    print(f"  Actual Died:        {cm[1,0]:5d}        {cm[1,1]:5d}")

    f1_macro = f1_score(y_te, y_pred, average='macro')
    f1_died  = f1_score(y_te, y_pred, pos_label=1)
    print(f"F1 (Died class)  : {f1_died:.4f}")
    print(f"F1 (macro avg)   : {f1_macro:.4f}")

    auc = None
    if len(np.unique(y_te)) > 1 and y_proba is not None:
        auc = roc_auc_score(y_te, y_proba)
        print(f"ROC-AUC          : {auc:.4f}")

    avg_conf = None
    if y_proba is not None:
        # "confidence" = mean predicted probability assigned to the predicted class
        conf_per_sample = np.where(y_pred == 1, y_proba, 1 - y_proba)
        avg_conf = conf_per_sample.mean()
        print(f"Avg confidence   : {avg_conf*100:.1f}%  "
              f"(mean predicted-class probability)")

    return {
        "confusion_matrix": cm,
        "f1_died": f1_died,
        "f1_macro": f1_macro,
        "auc": auc,
        "avg_confidence": avg_conf,
    }


# ─────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}\n")

    X, y, groups, pid_to_idx, pid_labels = load_all_data(
        DATA_ROOT, WINDOW_SIZE, STEP_SIZE
    )

    # ── PATIENT-WISE split via GroupShuffleSplit (no patient appears in both train & test) ──
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    # carve a val split out of train, also patient-wise, for CNN early stopping
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    tr_idx_rel, val_idx_rel = next(gss2.split(
        X[train_idx], y[train_idx], groups=groups[train_idx]
    ))
    tr_idx  = train_idx[tr_idx_rel]
    val_idx = train_idx[val_idx_rel]

    X_tr,  y_tr  = X[tr_idx],  y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_train, y_train = X[train_idx], y[train_idx]   # full train (tr+val) for feature extraction
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"Train patients : {len(np.unique(groups[tr_idx]))}  "
          f"| Val patients : {len(np.unique(groups[val_idx]))}  "
          f"| Test patients : {len(np.unique(groups[test_idx]))}")
    print(f"Train windows (CNN fit) : {len(y_tr):,}")
    print(f"Val   windows           : {len(y_val):,}")
    print(f"Full-train windows      : {len(y_train):,}")
    print(f"Test  windows           : {len(y_test):,}\n")

    idx_to_pid = {v: k for k, v in pid_to_idx.items()}
    test_pids  = np.unique(groups[test_idx])
    print(f'Test patients ({len(test_pids)} total):')
    print(f"  {'Idx':>6}  {'Patient ID':>12}  {'Label'}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}")
    for pidx in test_pids:
        real_pid  = idx_to_pid[pidx]
        lbl       = pid_labels[real_pid]
        lbl_str   = 'Died (1)' if lbl == 1 else 'Survived (0)'
        print(f'  {pidx:>6}  {real_pid:>12}  {lbl_str}')
    print()

    # Class weights for CNN
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)
    pos_weight = float(cw[1] / cw[0])
    print(f"Class weights  : survived={cw[0]:.3f}, died={cw[1]:.3f}")
    print(f"pos_weight     : {pos_weight:.3f}\n")

    # DataLoaders
    train_loader      = DataLoader(EEGDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader        = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader        = DataLoader(EEGDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    full_train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build & train CNN feature extractor
    model = EEG_CNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {n_params:,}\n")

    train_cnn(model, train_loader, val_loader, DEVICE,
              epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
              pos_weight=pos_weight, patience=PATIENCE)

    # Extract window-level CNN features (128-dim) for train/test
    print("── Phase 2: Extracting CNN features (window-level) ──")
    X_tr_feat, y_tr_feat = extract_features(model, full_train_loader, DEVICE)
    X_te_feat, y_te_feat = extract_features(model, test_loader,        DEVICE)
    print(f"  Train window features : {X_tr_feat.shape}")
    print(f"  Test  window features : {X_te_feat.shape}\n")

    torch.save(model.state_dict(), SAVE_MODEL)

    # ── PCA fit on train features only ──
    print(f"── Fitting PCA (n={PCA_COMPONENTS}) ───────")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_feat)
    X_te_pca = pca.transform(X_te_feat)
    var = pca.explained_variance_ratio_.sum() * 100
    print(f"  Explained variance : {var:.1f}%\n")
    joblib.dump(pca, SAVE_PCA)

    results = {}

    # ── Helper to run a given (raw / pca) feature set through all 3 classifiers ──
    def run_classifiers(tag, X_tr_, y_tr_, X_te_, y_te_):
        # SVM (calibrated for proba)
        print(f"\n========== {tag}: SVM ==========")
        base_svm = SVC(kernel='rbf', C=10.0, gamma='scale',
                        class_weight='balanced', random_state=42)
        svm = CalibratedClassifierCV(base_svm, cv=5, ensemble=False)
        svm.fit(X_tr_, y_tr_)
        y_pred = svm.predict(X_te_)
        y_proba = svm.predict_proba(X_te_)[:, 1]
        results[f"{tag}_SVM"] = evaluate(f"{tag} — SVM", y_te_, y_pred, y_proba)
        joblib.dump(svm, SAVE_DIR / f"svm_{tag}.pkl")

        # Random Forest
        print(f"\n========== {tag}: Random Forest ==========")
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf.fit(X_tr_, y_tr_)
        y_pred = rf.predict(X_te_)
        y_proba = rf.predict_proba(X_te_)[:, 1]
        results[f"{tag}_RF"] = evaluate(f"{tag} — Random Forest", y_te_, y_pred, y_proba)
        joblib.dump(rf, SAVE_DIR / f"rf_{tag}.pkl")

        # One-layer NN
        print(f"\n========== {tag}: One-Layer NN ==========")
        nn_model, y_pred, y_proba = train_one_layer_nn(
            X_tr_, y_tr_, X_te_, y_te_, DEVICE
        )
        results[f"{tag}_NN"] = evaluate(f"{tag} — One-Layer NN", y_te_, y_pred, y_proba)
        torch.save(nn_model.state_dict(), SAVE_DIR / f"onelayernn_{tag}.pth")

    # Without PCA (raw 128-dim CNN features)
    run_classifiers("RAW", X_tr_feat, y_tr_feat, X_te_feat, y_te_feat)

    # With PCA
    run_classifiers("PCA", X_tr_pca, y_tr_feat, X_te_pca, y_te_feat)

    # ── Summary ──
    print("\n\n========== SUMMARY (window-level test set) ==========")
    header = f"  {'Model':<14}{'F1(Died)':>10}{'F1(macro)':>11}{'AUC':>8}{'AvgConf':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k, r in results.items():
        f1d   = f"{r['f1_died']:.4f}"   if r['f1_died']   is not None else "N/A"
        f1m   = f"{r['f1_macro']:.4f}"  if r['f1_macro']  is not None else "N/A"
        auc   = f"{r['auc']:.4f}"       if r['auc']       is not None else "N/A"
        conf  = f"{r['avg_confidence']*100:.1f}%" if r['avg_confidence'] is not None else "N/A"
        print(f"  {k:<14}{f1d:>10}{f1m:>11}{auc:>8}{conf:>9}")

    print("\n── Confusion matrices ──")
    for k, r in results.items():
        cm = r["confusion_matrix"]
        print(f"\n  {k}")
        print(f"                 Pred:Survived  Pred:Died")
        print(f"  Actual Survived:    {cm[0,0]:5d}        {cm[0,1]:5d}")
        print(f"  Actual Died:        {cm[1,0]:5d}        {cm[1,1]:5d}")

    print(f"\nSaved: {SAVE_MODEL}, {SAVE_PCA}, and classifiers in {SAVE_DIR}/")


if __name__ == "__main__":
    main()