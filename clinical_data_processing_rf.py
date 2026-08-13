import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLINICAL_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data.xls"
EEG_CSV_DIR = r"C:\abalaji\bichat\eeg_csv"
LABELED_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data_labeled.csv"
FILTERED_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data_filtered.csv"
OUTPUT_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data_labeled_imputed.csv"
TARGET_COLS = ["IMC"]

DO_IMPUTATION = False
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

if Path(LABELED_PATH).exists() and Path(FILTERED_PATH).exists():
    print(f"{LABELED_PATH} and {FILTERED_PATH} already exist, skipping labeling step")
    df = pd.read_csv(LABELED_PATH)
    complete_rows = pd.read_csv(FILTERED_PATH)
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

    complete_rows = df.dropna().reset_index(drop=True)
    print(f"{len(complete_rows)} rows have no missing values across all columns:")
    print(complete_rows["N_PAT"].tolist())

    complete_rows = complete_rows.drop(columns=["N_PAT"])
    complete_rows.to_csv(FILTERED_PATH, index=False)
    print(f"Saved to {FILTERED_PATH}")

    df = df.drop(columns=["N_PAT"])
    df.to_csv(LABELED_PATH, index=False)
    print(f"Saved to {LABELED_PATH}")

if DO_IMPUTATION:
    if Path(OUTPUT_PATH).exists():
        print(f"{OUTPUT_PATH} already exists, skipping imputation")
        df = pd.read_csv(OUTPUT_PATH)
    else:
        # Count missing IMC before imputation
        before_missing = df["IMC"].isna().sum()

        # Mean IMC by obesity status
        mean_imc_obese = df.loc[df["Obésité"] == 1, "IMC"].mean()
        mean_imc_non_obese = df.loc[df["Obésité"] == 0, "IMC"].mean()

        print(f"Mean IMC (Obésité=1): {mean_imc_obese:.2f}")
        print(f"Mean IMC (Obésité=0): {mean_imc_non_obese:.2f}")

        # Impute IMC according to obesity status
        df.loc[df["IMC"].isna() & (df["Obésité"] == 1), "IMC"] = mean_imc_obese
        df.loc[df["IMC"].isna() & (df["Obésité"] == 0), "IMC"] = mean_imc_non_obese

        after_missing = df["IMC"].isna().sum()

        print(f"Imputed {before_missing - after_missing} IMC values")
        print(f"Remaining missing IMC: {after_missing}")

        # Remove rows with missing NSE
        before_rows = len(df)
        df = df[df["Valeur NSE 1 (µg/L)"].notna()].reset_index(drop=True)

        print(f"Removed {before_rows - len(df)} rows with missing NSE")
        print(f"Final dataset size: {len(df)} rows")

        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved to {OUTPUT_PATH}")

if DO_TRAIN_RF:

    rf_source = df if DO_IMPUTATION else complete_rows

    rf_df = rf_source.copy()

    rf_df["SEXE"] = LabelEncoder().fit_transform(rf_df["SEXE"])

    X = rf_df.drop(columns=["label"])
    y = rf_df["label"]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    from sklearn.model_selection import LeaveOneOut
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score
    )

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

        clf.fit(
            X_train,
            y_train
        )


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


    print(
        "\nAccuracy:",
        accuracy_score(
            all_true,
            all_pred
        )
    )


    print(
        "\nClassification Report"
    )
    print(
        classification_report(
            all_true,
            all_pred
        )
    )


    print(
        "\nConfusion Matrix"
    )
    print(
        confusion_matrix(
            all_true,
            all_pred
        )
    )


    if len(np.unique(all_true)) == 2:
        print(
            "\nROC-AUC:",
            roc_auc_score(
                all_true,
                all_prob_1
            )
        )


    results = rf_df.copy()

    results["True_Label"] = all_true
    results["Predicted_Label"] = all_pred

    results["P(Label=0)"] = all_prob_0
    results["P(Label=1)"] = all_prob_1


    results.to_csv(
        "randomforest_LOO_predictions.csv",
        index=False
    )


    print(
        "\nSaved: randomforest_LOO_predictions.csv"
    )