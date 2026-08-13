import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


FILTERED_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data_filtered.csv"


df = pd.read_csv(FILTERED_PATH)


nse_col = "Valeur NSE 1 (µg/L)"


df[nse_col] = np.log1p(df[nse_col])


df["AGE_NSE"] = (
    df["AGE"] *
    df[nse_col]
)


X = df.drop(columns=["label"])
y = df["label"]


X = X.reset_index(drop=True)
y = y.reset_index(drop=True)


continuous_cols = [
    "AGE",
    "IMC",
    "Valeur NSE 1 (µg/L)",
    "AGE_NSE"
]


binary_cols = [
    "SEXE",
    "HTA",
    "Diabète",
    "Obésité"
]


X_cont = X[continuous_cols]
X_bin = X[binary_cols]


scaler = StandardScaler()

X_cont_scaled = pd.DataFrame(
    scaler.fit_transform(X_cont),
    columns=continuous_cols
)


X_processed = pd.concat(
    [
        X_cont_scaled,
        X_bin.reset_index(drop=True)
    ],
    axis=1
)


loo = LeaveOneOut()


true_labels = []
prob_dead_nb = []
prob_dead_rf = []


for train_idx, test_idx in loo.split(X_processed):

    X_train = X_processed.iloc[train_idx]
    X_test = X_processed.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    X_train_cont = X_train[continuous_cols]
    X_test_cont = X_test[continuous_cols]

    X_train_bin = X_train[binary_cols]
    X_test_bin = X_test[binary_cols]


    gaussian = GaussianNB(
        priors=[0.5,0.5]
    )


    gaussian_params = {
        "var_smoothing": np.logspace(
            -12,
            0,
            20
        )
    }


    gaussian_search = GridSearchCV(
        gaussian,
        gaussian_params,
        scoring="roc_auc",
        cv=5
    )


    gaussian_search.fit(
        X_train_cont,
        y_train
    )


    gaussian = gaussian_search.best_estimator_


    bernoulli = BernoulliNB(
        alpha=1.0
    )


    bernoulli.fit(
        X_train_bin,
        y_train
    )


    p_gaussian = gaussian.predict_proba(
        X_test_cont
    )


    p_bernoulli = bernoulli.predict_proba(
        X_test_bin
    )


    p_dead = (
        p_gaussian[:,1] *
        p_bernoulli[:,1]
    )


    p_alive = (
        p_gaussian[:,0] *
        p_bernoulli[:,0]
    )


    total = p_alive + p_dead

    p_dead = p_dead / total


    true_labels.append(
        y_test.values[0]
    )

    prob_dead_nb.append(
        p_dead[0]
    )


    print(
        f"Patient {len(true_labels)} "
        f"True={y_test.values[0]} "
        f"P(Dead)={p_dead[0]:.3f}"
    )



print("\n==============================")
print("HYBRID BAYESIAN MODEL")
print("==============================")


print(
    "\nROC-AUC:",
    roc_auc_score(
        true_labels,
        prob_dead_nb
    )
)



threshold_results = []

for threshold in np.arange(0.05,0.65,0.01):

    preds = (
        np.array(prob_dead_nb)
        >= threshold
    ).astype(int)


    report = classification_report(
        true_labels,
        preds,
        output_dict=True
    )


    threshold_results.append(
        [
            threshold,
            report["1"]["precision"],
            report["1"]["recall"],
            report["1"]["f1-score"],
            accuracy_score(
                true_labels,
                preds
            )
        ]
    )


threshold_df = pd.DataFrame(
    threshold_results,
    columns=[
        "Threshold",
        "Dead_Precision",
        "Dead_Recall",
        "Dead_F1",
        "Accuracy"
    ]
)


print(
    threshold_df.sort_values(
        "Dead_F1",
        ascending=False
    ).head(10)
)

print("\nThreshold optimization")
print(threshold_df)



best_threshold = threshold_df.loc[
    threshold_df["Dead_F1"].idxmax(),
    "Threshold"
]


final_pred = (
    np.array(prob_dead_nb)
    >= best_threshold
).astype(int)


print(
    "\nBest threshold:",
    best_threshold
)


print(
    "\nClassification Report"
)

print(
    classification_report(
        true_labels,
        final_pred
    )
)


print(
    "\nConfusion Matrix"
)

print(
    confusion_matrix(
        true_labels,
        final_pred
    )
)



rf_predictions = []


for train_idx, test_idx in loo.split(X_processed):

    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42
    )


    rf.fit(
        X_processed.iloc[train_idx],
        y.iloc[train_idx]
    )


    p = rf.predict_proba(
        X_processed.iloc[test_idx]
    )


    rf_predictions.append(
        p[0][1]
    )



print("\n==============================")
print("RANDOM FOREST LOO")
print("==============================")


print(
    "ROC-AUC:",
    roc_auc_score(
        true_labels,
        rf_predictions
    )
)



results = X.copy()

results["True_Label"] = true_labels
results["Bayesian_P_Dead"] = prob_dead_nb
results["Bayesian_Pred"] = final_pred
results["RF_P_Dead"] = rf_predictions


results.to_csv(
    "hybrid_bayesian_vs_RF_LOO.csv",
    index=False
)


print(
    "\nSaved hybrid_bayesian_vs_RF_LOO.csv"
)