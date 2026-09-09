import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

CLINICAL_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data.xls"
EEG_CSV_DIR = r"C:\abalaji\bichat\eeg_csv"
LABELED_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data_labeled.csv"
FILTERED_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data_filtered.csv"
IMPUTED_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data_labeled_imputed.csv"

DO_IMPUTATION = True
DO_TRAIN_RF = True

COLUMNS = [
    "N_PAT",
    "SEXE",
    "AGE",
    "HTA",
    "Diabète",
    "Obésité",
    "IMC",
    "Valeur NSE 1 (µg/L)",
]

BINARY_COLS = ["SEXE", "HTA", "Diabète", "Obésité"]

def find_label(pat_id):
    pat_id = str(int(pat_id))
    for label in [0, 1]:
        folder = Path(EEG_CSV_DIR) / str(label)
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.stem.startswith(pat_id):
                return label
    return None

if Path(LABELED_PATH).exists():
    print(f"{LABELED_PATH} already exists, skipping labeling step")
    df = pd.read_csv(LABELED_PATH)
else:
    df = pd.read_excel(CLINICAL_PATH)
    df = df[COLUMNS]
    df = df[df["N_PAT"].notna()].reset_index(drop=True)
    df["label"] = df["N_PAT"].apply(find_label)

    missing = df[df["label"].isna()]
    if len(missing) > 0:
        print(f"{len(missing)} patients not found in eeg_csv/0 or /1:")
        print(missing["N_PAT"].tolist())

    df = df[df["label"].notna()].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    df = df.drop(columns=["N_PAT"])
    df.to_csv(LABELED_PATH, index=False)
    print(f"Saved to {LABELED_PATH}")
    print(f"{len(df)} patients labeled")

if Path(FILTERED_PATH).exists():
    print(f"{FILTERED_PATH} already exists, skipping filtering step")
    complete_rows = pd.read_csv(FILTERED_PATH)
else:
    complete_rows = df.dropna().reset_index(drop=True)
    print(f"{len(complete_rows)} of {len(df)} patients have no missing values across all columns")
    complete_rows.to_csv(FILTERED_PATH, index=False)
    print(f"Saved to {FILTERED_PATH}")

if DO_IMPUTATION:
    if Path(IMPUTED_PATH).exists():
        print(f"{IMPUTED_PATH} already exists, skipping imputation")
        df_imputed = pd.read_csv(IMPUTED_PATH)
    else:
        df_imputed = df.copy()

        sexe_encoder = LabelEncoder()
        non_null_sexe = df_imputed["SEXE"].dropna()
        sexe_encoder.fit(non_null_sexe)
        df_imputed["SEXE"] = df_imputed["SEXE"].map(
            lambda v: sexe_encoder.transform([v])[0] if pd.notna(v) else np.nan
        )

        feature_cols = [c for c in df_imputed.columns if c != "label"]
        missing_before = df_imputed[feature_cols].isna().sum()
        print("Missing values before imputation:")
        print(missing_before[missing_before > 0])

        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, random_state=42),
            random_state=42,
            max_iter=10
        )
        imputed_array = imputer.fit_transform(df_imputed[feature_cols])
        df_imputed[feature_cols] = imputed_array

        for col in BINARY_COLS:
            n_classes = len(sexe_encoder.classes_) if col == "SEXE" else 2
            df_imputed[col] = df_imputed[col].round().clip(0, n_classes - 1).astype(int)

        df_imputed["SEXE"] = sexe_encoder.inverse_transform(df_imputed["SEXE"])

        missing_after = df_imputed[feature_cols].isna().sum().sum()
        print(f"Remaining missing values after imputation: {missing_after}")

        df_imputed.to_csv(IMPUTED_PATH, index=False)
        print(f"Saved to {IMPUTED_PATH}")

if DO_TRAIN_RF:

    rf_source = df_imputed if DO_IMPUTATION else complete_rows

    rf_df = rf_source.copy()
    rf_df["SEXE"] = LabelEncoder().fit_transform(rf_df["SEXE"])

    X = rf_df.drop(columns=["label"]).reset_index(drop=True)
    y = rf_df["label"].reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    loo = LeaveOneOut()

    all_true = []
    all_pred = []
    all_prob_0 = []
    all_prob_1 = []

    counter = 1

    for train_index, test_index in loo.split(X_scaled):

        X_train = X_scaled.iloc[train_index]
        X_test = X_scaled.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42
        )

        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)

        all_true.append(y_test.values[0])
        all_pred.append(pred[0])
        all_prob_0.append(prob[0][0])
        all_prob_1.append(prob[0][1])

        print(
            f"Patient {counter}/{len(X_scaled)} "
            f"True={y_test.values[0]} "
            f"Pred={pred[0]} "
            f"P0={prob[0][0]:.3f} "
            f"P1={prob[0][1]:.3f}"
        )

        counter += 1

    print("\n==============================")
    print("LOO RANDOM FOREST RESULTS")
    print("==============================")

    print("\nAccuracy:", accuracy_score(all_true, all_pred))
    print("\nClassification Report")
    print(classification_report(all_true, all_pred))
    print("\nConfusion Matrix")
    print(confusion_matrix(all_true, all_pred))

    if len(np.unique(all_true)) == 2:
        print("\nROC-AUC:", roc_auc_score(all_true, all_prob_1))
