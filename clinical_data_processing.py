import pandas as pd
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

CLINICAL_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data.xls"
EEG_CSV_DIR = r"C:\abalaji\bichat\eeg_csv"
LABELED_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data_labeled.xlsx"
OUTPUT_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data_labeled_imputed.xlsx"
TARGET_COLS = ["Valeur NSE 1 (µg/L)", "Valeur NSE 2 (µg/L)"]

def find_label(pat_id):
    pat_id = str(pat_id)
    for label in [0, 1]:
        folder = Path(EEG_CSV_DIR) / str(label)
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.stem.startswith(pat_id):
                return label
    return None

if not Path(LABELED_PATH).exists():
    df = pd.read_excel(CLINICAL_PATH)
    df["label"] = df["N_PAT"].apply(find_label)

    missing = df[df["label"].isna()]
    if len(missing) > 0:
        print(f"{len(missing)} patients not found in eeg_csv/0 or /1:")
        print(missing["N_PAT"].tolist())

    df.to_excel(LABELED_PATH, index=False)
    print(f"Saved to {LABELED_PATH}")
else:
    print(f"{LABELED_PATH} already exists, skipping labeling")
    df = pd.read_excel(LABELED_PATH)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
for col in TARGET_COLS:
    if col not in numeric_cols:
        numeric_cols.append(col)

before_missing = {col: df[col].isna().sum() for col in TARGET_COLS}

imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    max_iter=10,
    random_state=42
)

imputed_array = imputer.fit_transform(df[numeric_cols])
imputed_df = pd.DataFrame(imputed_array, columns=numeric_cols, index=df.index)

for col in TARGET_COLS:
    df[col] = imputed_df[col]

for col in TARGET_COLS:
    after_missing = df[col].isna().sum()
    print(f"Imputed {before_missing[col] - after_missing} missing values in '{col}'")
    print(f"Remaining missing: {after_missing}")

df.to_excel(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")