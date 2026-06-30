"""
EEG Death vs No-Death Classification — v5 (lowpass-filtered, non-overlapping holdout window split, multi-classifier)
CNN → Flatten/FC(128) features → {SVM, RandomForest, 1-layer NN} × {with PCA, without PCA}

Changes vs v4:
  - NEW: Butter lowpass filtering preprocessing step.
        butter_lowpass_filter(signal, order, cutoff, fs) — filters a single channel.
        filter_and_save_all_files(...) — walks every CSV in DATA_ROOT, applies the
        lowpass filter to each of the 16 EEG channels, and saves the filtered CSVs
        to FILTERED_DATA_ROOT, mirroring the 0/ and 1/ folder structure.
        This runs once (skipped if output already exists, unless FORCE_REFILTER=True)
        before any windowing/training, and all downstream loading reads from
        FILTERED_DATA_ROOT instead of the raw DATA_ROOT.
  - Everything else (per-chunk holdout test window, non-overlapping train windows,
    CNN feature extractor, 6 classifier variants) is unchanged from v4.

Folder structure (raw):
  chunks_20/0/  ← survived  (1214_chunk_000.csv ...)
  chunks_20/1/  ← died      (2045_chunk_000.csv ...)
Each CSV: (15361, 17) — header row + 16 EEG cols + 1 extra col (dropped)

Folder structure (filtered, auto-created):
  chunks_20_filtered/0/ ...
  chunks_20_filtered/1/ ...
"""

import re
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import defaultdict, Counter

from scipy.signal import butter, filtfilt, freqz

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score
)

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
DATA_ROOT          = Path(r"D:\abalaji\chunks_20")
FILTERED_DATA_ROOT  = Path(r"D:\abalaji\chunks_20_filtered")
FORCE_REFILTER      = True

N_CHANNELS     = 16
WINDOW_SIZE    = 256
STEP_SIZE      = 128
BATCH_SIZE     = 64
EPOCHS         = 25
LR             = 5e-4
WEIGHT_DECAY   = 1e-4
PCA_COMPONENTS = 32
PATIENCE       = 7
VAL_SIZE       = 0.15
RANDOM_SEED    = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lowpass filter settings
FILTER_ORDER   = 4
FILTER_CUTOFF  = 40.0     # Hz
FILTER_FS      = 256.0    # Hz — sampling rate of the EEG signal; adjust to your actual fs

SAVE_MODEL     = "eeg_cnn_v5.pth"
SAVE_PCA       = "eeg_pca_v5.pkl"
SAVE_DIR       = Path("models_v5")
SAVE_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(RANDOM_SEED)   # controls which window is held out per chunk


# ─────────────────────────────────────────────────────────
# 0. LOWPASS FILTERING
# ─────────────────────────────────────────────────────────
def butter_lowpass_filter(signal, order, cutoff, fs):
    signal = pd.to_numeric(signal, errors='coerce').to_numpy(dtype=np.float64) \
        if isinstance(signal, pd.Series) else np.asarray(signal, dtype=np.float64)
    if np.isnan(signal).any():
        raise ValueError(
            f"Signal contains NaN after numeric coercion — "
            f"{np.isnan(signal).sum()} non-numeric value(s) found."
        )
    fs = int(fs)
    nyq = 0.5 * fs
    midcut = cutoff / nyq
    b, a = butter(order, midcut, btype="lowpass")
    w, h = freqz(b, a)
    return filtfilt(b, a, signal)


def filter_and_save_all_files(data_root, output_root, order, cutoff, fs,
                               n_channels=N_CHANNELS, force=False):
    """
    Walk every CSV under data_root/{0,1}/, lowpass-filter each of the first
    n_channels columns independently, and write the filtered CSV (same shape,
    same header, same filename) into output_root/{0,1}/.

    Skips the whole pass if output_root already contains files for both
    classes, unless force=True.
    """
    output_root = Path(output_root)

    already_done = (
        output_root.exists()
        and any((output_root / "0").glob("*.csv"))
        and any((output_root / "1").glob("*.csv"))
    )
    if already_done and not force:
        print(f"[Filtering] {output_root} already populated — skipping "
              f"(set FORCE_REFILTER=True to redo).\n")
        return

    print(f"[Filtering] Lowpass filtering all CSVs in {data_root} "
          f"(order={order}, cutoff={cutoff}Hz, fs={fs}Hz) → {output_root}")

    n_files = 0
    n_failed = 0
    for label in [0, 1]:
        in_folder  = Path(data_root) / str(label)
        out_folder = output_root / str(label)
        out_folder.mkdir(parents=True, exist_ok=True)

        if not in_folder.exists():
            raise FileNotFoundError(f"Folder not found: {in_folder}")

        csv_files = sorted(in_folder.glob("*.csv"))
        print(f"  /{label}: {len(csv_files)} files")

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, header=0)
                ch_cols = df.columns[:n_channels]

                filtered_df = df.copy()
                for col in ch_cols:
                    filtered_df[col] = butter_lowpass_filter(
                        df[col], order=order, cutoff=cutoff, fs=fs
                    )

                out_path = out_folder / csv_path.name
                filtered_df.to_csv(out_path, index=False)
                n_files += 1
            except Exception as e:
                n_failed += 1
                print(f"    [FAIL] {csv_path.name}: {e}")

        if (n_files) % 50 == 0:
            print(f"    ...{n_files} files filtered so far")

    print(f"[Filtering] Done. {n_files} files filtered, {n_failed} failed.\n")


# ─────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────
def parse_filename(filename):
    match = re.match(r"(\d+)_chunk_(\d+)\.csv", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def extract_windows_with_holdout(chunk, label, window_size, step_size, rng):
    """
    Pick ONE random window position in `chunk` to be a TEST window.
    Build TRAIN windows via the normal sliding-window scheme over the
    rest of the chunk, skipping any training window whose span overlaps
    the test window's span (so train/test never share raw samples from
    this chunk).
    """
    n = len(chunk)
    max_start = n - window_size

    if max_start < 0:
        return (np.empty((0, window_size, chunk.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                None, None)

    test_start = int(rng.integers(0, max_start + 1))
    test_end   = test_start + window_size
    test_window = chunk[test_start:test_end].astype(np.float32)

    train_windows = []
    for start in range(0, max_start + 1, step_size):
        end = start + window_size
        if end <= test_start or start >= test_end:
            train_windows.append(chunk[start:end])

    if train_windows:
        train_windows = np.array(train_windows, dtype=np.float32)
    else:
        train_windows = np.empty((0, window_size, chunk.shape[1]), dtype=np.float32)

    train_labels = np.full(len(train_windows), label, dtype=np.int64)

    return train_windows, train_labels, test_window, label


def load_all_data(data_root, window_size, step_size, rng):
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

    train_windows_all, train_labels_all, train_groups_all = [], [], []
    test_windows_all,  test_labels_all,  test_groups_all  = [], [], []
    skipped = 0
    n_chunks_used = 0

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

            tr_w, tr_l, te_w, te_l = extract_windows_with_holdout(
                chunk, label, window_size, step_size, rng
            )
            n_chunks_used += 1

            if len(tr_w) > 0:
                train_windows_all.append(tr_w)
                train_labels_all.append(tr_l)
                train_groups_all.extend([pid_idx] * len(tr_w))

            if te_w is not None:
                test_windows_all.append(te_w[np.newaxis, ...])
                test_labels_all.append(te_l)
                test_groups_all.append(pid_idx)

        if (pid_idx + 1) % 20 == 0:
            print(f"  Processed {pid_idx+1}/{len(all_pids)} patients...")

    print(f"Skipped files : {skipped}")
    print(f"Chunks contributing a holdout test window : {n_chunks_used}")

    X_train = np.concatenate(train_windows_all, axis=0)[..., np.newaxis]
    y_train = np.concatenate(train_labels_all,  axis=0)
    g_train = np.array(train_groups_all)

    X_test  = np.concatenate(test_windows_all, axis=0)[..., np.newaxis]
    y_test  = np.array(test_labels_all, dtype=np.int64)
    g_test  = np.array(test_groups_all)

    print(f"\n── Dataset Summary ──────────────────")
    print(f"Train windows  : {X_train.shape}  "
          f"(died={int((y_train==1).sum()):,}, survived={int((y_train==0).sum()):,})")
    print(f"Test  windows  : {X_test.shape}  "
          f"(died={int((y_test==1).sum()):,}, survived={int((y_test==0).sum()):,})")
    print(f"─────────────────────────────────────\n")

    return X_train, y_train, g_train, X_test, y_test, g_test, pid_to_idx, pid_labels


# ─────────────────────────────────────────────────────────
# 2. PYTORCH DATASET
# ─────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, X, y):
        X_t = np.transpose(X, (0, 3, 1, 2))
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
            ConvBlock(1,  32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
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
        return x

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
# 7. EVALUATION
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

    # ── Phase 0: Lowpass-filter every raw CSV and save into FILTERED_DATA_ROOT ──
    filter_and_save_all_files(
        DATA_ROOT, FILTERED_DATA_ROOT,
        order=FILTER_ORDER, cutoff=FILTER_CUTOFF, fs=FILTER_FS,
        force=FORCE_REFILTER
    )

    # ── Load data (now reading from the FILTERED folder): per-chunk holdout
    #    test window + non-overlapping train windows ──
    (X_train, y_train, g_train,
     X_test,  y_test,  g_test,
     pid_to_idx, pid_labels) = load_all_data(FILTERED_DATA_ROOT, WINDOW_SIZE, STEP_SIZE, RNG)

    # ── Carve a val split out of the TRAIN window pool only (window-wise stratified) ──
    idx = np.arange(len(y_train))
    tr_idx, val_idx = train_test_split(
        idx, test_size=VAL_SIZE, random_state=0, stratify=y_train
    )

    X_tr,  y_tr  = X_train[tr_idx],  y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    print(f"Train windows (CNN fit) : {len(y_tr):,}")
    print(f"Val   windows           : {len(y_val):,}")
    print(f"Full-train windows      : {len(y_train):,}")
    print(f"Test  windows           : {len(y_test):,}\n")

    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)
    pos_weight = float(cw[1] / cw[0])
    print(f"Class weights  : survived={cw[0]:.3f}, died={cw[1]:.3f}")
    print(f"pos_weight     : {pos_weight:.3f}\n")

    train_loader      = DataLoader(EEGDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader        = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader       = DataLoader(EEGDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    full_train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = EEG_CNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {n_params:,}\n")

    train_cnn(model, train_loader, val_loader, DEVICE,
              epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
              pos_weight=pos_weight, patience=PATIENCE)

    print("── Phase 2: Extracting CNN features (window-level) ──")
    X_tr_feat, y_tr_feat = extract_features(model, full_train_loader, DEVICE)
    X_te_feat, y_te_feat = extract_features(model, test_loader,        DEVICE)
    print(f"  Train window features : {X_tr_feat.shape}")
    print(f"  Test  window features : {X_te_feat.shape}\n")

    torch.save(model.state_dict(), SAVE_MODEL)

    print(f"── Fitting PCA (n={PCA_COMPONENTS}) ───────")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_feat)
    X_te_pca = pca.transform(X_te_feat)
    var = pca.explained_variance_ratio_.sum() * 100
    print(f"  Explained variance : {var:.1f}%\n")
    joblib.dump(pca, SAVE_PCA)

    results = {}

    def run_classifiers(tag, X_tr_, y_tr_, X_te_, y_te_):
        print(f"\n========== {tag}: SVM ==========")
        base_svm = SVC(kernel='rbf', C=10.0, gamma='scale',
                        class_weight='balanced', random_state=42)
        svm = CalibratedClassifierCV(base_svm, cv=5, ensemble=False)
        svm.fit(X_tr_, y_tr_)
        y_pred = svm.predict(X_te_)
        y_proba = svm.predict_proba(X_te_)[:, 1]
        results[f"{tag}_SVM"] = evaluate(f"{tag} — SVM", y_te_, y_pred, y_proba)
        joblib.dump(svm, SAVE_DIR / f"svm_{tag}.pkl")

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

        print(f"\n========== {tag}: One-Layer NN ==========")
        nn_model, y_pred, y_proba = train_one_layer_nn(
            X_tr_, y_tr_, X_te_, y_te_, DEVICE
        )
        results[f"{tag}_NN"] = evaluate(f"{tag} — One-Layer NN", y_te_, y_pred, y_proba)
        torch.save(nn_model.state_dict(), SAVE_DIR / f"onelayernn_{tag}.pth")

    run_classifiers("RAW", X_tr_feat, y_tr_feat, X_te_feat, y_te_feat)
    run_classifiers("PCA", X_tr_pca, y_tr_feat, X_te_pca, y_te_feat)

    print("\n\n========== SUMMARY (held-out per-chunk test windows) ==========")
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