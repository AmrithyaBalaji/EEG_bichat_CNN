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

DATA_ROOT          = Path(r"D:\abalaji\chunks_20")
FILTERED_DATA_ROOT  = Path(r"D:\abalaji\chunks_20_filtered")
FORCE_REFILTER      = False

N_CHANNELS     = 16
WINDOW_SIZE    = 256
STEP_SIZE      = 256
BATCH_SIZE     = 64
EPOCHS         = 25
LR             = 5e-4
WEIGHT_DECAY   = 1e-4
PCA_COMPONENTS = 32
PATIENCE       = 7
VAL_SIZE       = 0.15
RANDOM_SEED    = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILTER_ORDER   = 4
FILTER_CUTOFF  = 40.0
FILTER_FS      = 256.0

N_SURVIVED_TEST_PER_FOLD = 3
N_DIED_TEST_PER_FOLD     = 1

SAVE_DIR       = Path("models_v5_kfold")
SAVE_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(RANDOM_SEED)

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

def parse_filename(filename):
    match = re.match(r"(\d+)_chunk_(\d+)\.csv", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def discover_patient_files(data_root):
    patient_files = defaultdict(list)

    for label in [0, 1]:
        folder = Path(data_root) / str(label)
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

    pid_labels = {pid: chunks[0][2] for pid, chunks in patient_files.items()}
    lc = Counter(pid_labels.values())
    print(f"\nTotal unique patients : {len(patient_files)}")
    print(f"  Survived (0) : {lc[0]} patients")
    print(f"  Died     (1) : {lc[1]} patients\n")

    return patient_files, pid_labels

def make_patient_folds(pid_labels, rng,
                        n_survived_per_fold=N_SURVIVED_TEST_PER_FOLD,
                        n_died_per_fold=N_DIED_TEST_PER_FOLD):
    died_pids     = [pid for pid, lbl in pid_labels.items() if lbl == 1]
    survived_pids = [pid for pid, lbl in pid_labels.items() if lbl == 0]

    if len(died_pids) < n_died_per_fold:
        raise ValueError(
            f"Need at least {n_died_per_fold} died patient(s) to form a fold; "
            f"found {len(died_pids)}."
        )
    if len(survived_pids) < n_survived_per_fold:
        raise ValueError(
            f"Need at least {n_survived_per_fold} survived patient(s) to form a fold; "
            f"found {len(survived_pids)}."
        )

    died_pids     = rng.permutation(np.array(died_pids)).tolist()
    survived_pids = rng.permutation(np.array(survived_pids)).tolist()

    n_folds = len(died_pids) // n_died_per_fold
    all_pids = set(pid_labels.keys())

    folds = []
    for i in range(n_folds):
        test_died = died_pids[i * n_died_per_fold:(i + 1) * n_died_per_fold]

        start = (i * n_survived_per_fold) % len(survived_pids)
        surv_idx = [(start + j) % len(survived_pids) for j in range(n_survived_per_fold)]
        test_survived = [survived_pids[j] for j in surv_idx]

        test_pids  = set(test_died) | set(test_survived)
        train_pids = all_pids - test_pids

        folds.append({
            "fold": i,
            "train_pids": sorted(train_pids),
            "test_pids":  sorted(test_pids),
            "test_died":  test_died,
            "test_survived": test_survived,
        })

    return folds

def extract_windows(chunk, label, window_size, step_size):
    n = len(chunk)
    max_start = n - window_size
    if max_start < 0:
        return (np.empty((0, window_size, chunk.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.int64))

    windows = []
    for start in range(0, max_start + 1, step_size):
        end = start + window_size
        windows.append(chunk[start:end])

    windows = np.array(windows, dtype=np.float32)
    labels  = np.full(len(windows), label, dtype=np.int64)
    return windows, labels

def build_windows_for_patients(patient_files, pids, window_size, step_size,
                                n_channels=N_CHANNELS, tag=""):
    pid_to_idx = {pid: idx for idx, pid in enumerate(sorted(patient_files.keys()))}

    windows_all, labels_all, groups_all = [], [], []
    skipped = 0
    n_chunks = 0

    for pid in pids:
        chunks = sorted(patient_files[pid], key=lambda x: x[0])
        for cid, csv_path, label in chunks:
            df = pd.read_csv(csv_path, header=0)
            chunk = df.values.astype(np.float32)

            if chunk.shape[1] == n_channels + 1:
                chunk = chunk[:, :n_channels]

            if chunk.shape[1] != n_channels or chunk.shape[0] < window_size:
                print(f"  [SKIP] {csv_path.name} shape: {chunk.shape}")
                skipped += 1
                continue

            scaler = StandardScaler()
            chunk  = scaler.fit_transform(chunk)

            w, l = extract_windows(chunk, label, window_size, step_size)
            n_chunks += 1
            if len(w) > 0:
                windows_all.append(w)
                labels_all.append(l)
                groups_all.extend([pid_to_idx[pid]] * len(w))

    if not windows_all:
        raise RuntimeError(f"No windows produced for {tag} set — check patient ids / data.")

    X = np.concatenate(windows_all, axis=0)[..., np.newaxis]
    y = np.concatenate(labels_all, axis=0)
    g = np.array(groups_all)

    print(f"  [{tag}] patients={len(pids)}  chunks_used={n_chunks}  skipped={skipped}  "
          f"windows={X.shape}  (died={int((y==1).sum())}, survived={int((y==0).sum())})")

    return X, y, g

class EEGDataset(Dataset):
    def __init__(self, X, y):
        X_t = np.transpose(X, (0, 3, 1, 2))
        self.X = torch.tensor(X_t, dtype=torch.float32)
        self.y = torch.tensor(y,   dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

def train_cnn(model, train_loader, val_loader, device,
              epochs, lr, weight_decay, pos_weight, patience, min_delta=1e-3,
              log_prefix=""):

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

    print(f"{log_prefix}── Training CNN ─────────────")
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

        print(f"{log_prefix}  Epoch [{epoch:02d}/{epochs}]  "
              f"Train Loss: {avg_tr_loss:.4f} Acc: {tr_acc:.1f}%  |  "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}%")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"{log_prefix}  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"{log_prefix}  Best val loss: {best_val_loss:.4f}\n")

def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            f = model.forward_features(X_b.to(device))
            feats.append(f.cpu().numpy())
            labels.append(y_b.numpy())
    return np.concatenate(feats), np.concatenate(labels)

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

def evaluate(name, y_te, y_pred, y_proba):
    print(f"\n── Results: {name} ────────────────────────")
    print(classification_report(
        y_te, y_pred,
        target_names=['Survived (0)', 'Died (1)'],
        zero_division=0
    ))
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                 Pred:Survived  Pred:Died")
    print(f"  Actual Survived:    {cm[0,0]:5d}        {cm[0,1]:5d}")
    print(f"  Actual Died:        {cm[1,0]:5d}        {cm[1,1]:5d}")

    f1_macro = f1_score(y_te, y_pred, average='macro', zero_division=0)
    f1_died  = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
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

def run_fold(fold_info, patient_files, device):
    fold_id     = fold_info["fold"]
    train_pids  = fold_info["train_pids"]
    test_pids   = fold_info["test_pids"]

    print(f"\n\n################## FOLD {fold_id} ##################")
    print(f"  Test patients (died)     : {fold_info['test_died']}")
    print(f"  Test patients (survived) : {fold_info['test_survived']}")
    print(f"  Train patients           : {len(train_pids)}")

    X_train, y_train, g_train = build_windows_for_patients(
        patient_files, train_pids, WINDOW_SIZE, STEP_SIZE, tag=f"fold{fold_id}-TRAIN"
    )
    X_test, y_test, g_test = build_windows_for_patients(
        patient_files, test_pids, WINDOW_SIZE, STEP_SIZE, tag=f"fold{fold_id}-TEST"
    )

    assert set(train_pids).isdisjoint(set(test_pids)), \
        "Leakage: a patient appears in both train and test pid lists!"

    idx = np.arange(len(y_train))
    tr_idx, val_idx = train_test_split(
        idx, test_size=VAL_SIZE, random_state=0, stratify=y_train
    )
    X_tr,  y_tr  = X_train[tr_idx],  y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)
    pos_weight = float(cw[1] / cw[0])

    train_loader      = DataLoader(EEGDataset(X_tr,  y_tr),  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader        = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader       = DataLoader(EEGDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    full_train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = EEG_CNN().to(device)
    train_cnn(model, train_loader, val_loader, device,
              epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
              pos_weight=pos_weight, patience=PATIENCE,
              log_prefix=f"[fold{fold_id}] ")

    X_tr_feat, y_tr_feat = extract_features(model, full_train_loader, device)
    X_te_feat, y_te_feat = extract_features(model, test_loader,        device)

    torch.save(model.state_dict(), SAVE_DIR / f"cnn_fold{fold_id}.pth")

    pca = PCA(n_components=min(PCA_COMPONENTS, X_tr_feat.shape[0], X_tr_feat.shape[1]),
              random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_feat)
    X_te_pca = pca.transform(X_te_feat)
    joblib.dump(pca, SAVE_DIR / f"pca_fold{fold_id}.pkl")

    fold_results = {}

    def run_classifiers(tag, X_tr_, y_tr_, X_te_, y_te_):
        print(f"\n===== fold{fold_id} / {tag}: SVM =====")
        base_svm = SVC(kernel='rbf', C=10.0, gamma='scale',
                        class_weight='balanced', random_state=42)
        svm = CalibratedClassifierCV(base_svm, cv=5, ensemble=False)
        svm.fit(X_tr_, y_tr_)
        y_pred = svm.predict(X_te_)
        y_proba = svm.predict_proba(X_te_)[:, 1]
        fold_results[f"{tag}_SVM"] = evaluate(f"fold{fold_id} {tag} — SVM", y_te_, y_pred, y_proba)

        print(f"\n===== fold{fold_id} / {tag}: Random Forest =====")
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf.fit(X_tr_, y_tr_)
        y_pred = rf.predict(X_te_)
        y_proba = rf.predict_proba(X_te_)[:, 1]
        fold_results[f"{tag}_RF"] = evaluate(f"fold{fold_id} {tag} — Random Forest", y_te_, y_pred, y_proba)

        print(f"\n===== fold{fold_id} / {tag}: One-Layer NN =====")
        nn_model, y_pred, y_proba = train_one_layer_nn(X_tr_, y_tr_, X_te_, y_te_, device)
        fold_results[f"{tag}_NN"] = evaluate(f"fold{fold_id} {tag} — One-Layer NN", y_te_, y_pred, y_proba)

    run_classifiers("RAW", X_tr_feat, y_tr_feat, X_te_feat, y_te_feat)
    run_classifiers("PCA", X_tr_pca, y_tr_feat, X_te_pca, y_te_feat)

    return fold_results

def main():
    print(f"Device : {DEVICE}\n")

    filter_and_save_all_files(
        DATA_ROOT, FILTERED_DATA_ROOT,
        order=FILTER_ORDER, cutoff=FILTER_CUTOFF, fs=FILTER_FS,
        force=FORCE_REFILTER
    )

    patient_files, pid_labels = discover_patient_files(FILTERED_DATA_ROOT)

    folds = make_patient_folds(
        pid_labels, RNG,
        n_survived_per_fold=N_SURVIVED_TEST_PER_FOLD,
        n_died_per_fold=N_DIED_TEST_PER_FOLD,
    )
    print(f"Built {len(folds)} fold(s); each test fold = "
          f"{N_SURVIVED_TEST_PER_FOLD} survived + {N_DIED_TEST_PER_FOLD} died patient(s), "
          f"fully excluded from that fold's training set.")

    all_fold_results = []
    for fold_info in folds:
        fold_results = run_fold(fold_info, patient_files, DEVICE)
        all_fold_results.append(fold_results)

    print("\n\n========== K-FOLD SUMMARY (mean ± std across folds) ==========")
    model_keys = sorted(all_fold_results[0].keys())
    header = f"  {'Model':<14}{'F1(Died)':>16}{'F1(macro)':>16}{'AUC':>16}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in model_keys:
        f1d = [r[k]['f1_died']  for r in all_fold_results if r[k]['f1_died']  is not None]
        f1m = [r[k]['f1_macro'] for r in all_fold_results if r[k]['f1_macro'] is not None]
        auc = [r[k]['auc']      for r in all_fold_results if r[k]['auc']      is not None]

        def fmt(vals):
            if not vals:
                return "N/A"
            return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

        print(f"  {k:<14}{fmt(f1d):>16}{fmt(f1m):>16}{fmt(auc):>16}")

    print(f"\nPer-fold models/artifacts saved in {SAVE_DIR}/")

if __name__ == "__main__":
    main()