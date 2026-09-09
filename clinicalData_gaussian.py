import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)


RUN_MODELS = {
    "hybrid_nb": False,
    "rf": False,
    "lr": True
}


FILTERED_PATH = r"C:\abalaji\bichat\EEG_bichat_CNN\clinical_data_filtered.csv"


df = pd.read_csv(FILTERED_PATH)


nse_col = "Valeur NSE 1 (µg/L)"


df[nse_col] = np.log1p(df[nse_col])

X = df.drop(columns=["label"])
y = df["label"]


X = X.reset_index(drop=True)
y = y.reset_index(drop=True)


continuous_cols = [
    "AGE",
    "Valeur NSE 1 (µg/L)",
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


def fit_hybrid_nb(X_train_cont, y_train, X_train_bin, priors):

    gaussian = GaussianNB(
        priors=priors
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

    return gaussian, bernoulli


def predict_hybrid_nb(gaussian, bernoulli, X_cont, X_bin):

    p_gaussian = gaussian.predict_proba(X_cont)
    p_bernoulli = bernoulli.predict_proba(X_bin)

    p_dead = p_gaussian[:, 1] * p_bernoulli[:, 1]
    p_alive = p_gaussian[:, 0] * p_bernoulli[:, 0]

    total = p_alive + p_dead

    return p_dead / total


def select_best_threshold(y_true, p_dead):

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.05, 0.65, 0.01):

        preds = (np.array(p_dead) >= threshold).astype(int)

        if preds.sum() == 0 or preds.sum() == len(preds):
            continue

        score = f1_score(y_true, preds, pos_label=1, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


loo = LeaveOneOut()

true_labels = []
prob_dead_nb = []
fold_thresholds = []
final_pred = []

rf_predictions = []
rf_preds = []

lr_predictions = []
lr_preds = []


if RUN_MODELS["hybrid_nb"]:

    for train_idx, test_idx in loo.split(X_processed):

        X_train = X_processed.iloc[train_idx]
        X_test = X_processed.iloc[test_idx]

        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx]

        X_train_cont = X_train[continuous_cols].reset_index(drop=True)
        X_test_cont = X_test[continuous_cols]

        X_train_bin = X_train[binary_cols].reset_index(drop=True)
        X_test_bin = X_test[binary_cols]

        class_counts = y_train.value_counts(normalize=True).sort_index()
        priors = [class_counts.get(0, 0.5), class_counts.get(1, 0.5)]

        inner_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        inner_true = []
        inner_prob = []

        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_cont, y_train):

            inner_gaussian, inner_bernoulli = fit_hybrid_nb(
                X_train_cont.iloc[inner_train_idx],
                y_train.iloc[inner_train_idx],
                X_train_bin.iloc[inner_train_idx],
                priors
            )

            inner_p_dead = predict_hybrid_nb(
                inner_gaussian,
                inner_bernoulli,
                X_train_cont.iloc[inner_val_idx],
                X_train_bin.iloc[inner_val_idx]
            )

            inner_true.extend(y_train.iloc[inner_val_idx].values)
            inner_prob.extend(inner_p_dead)

        fold_threshold = select_best_threshold(inner_true, inner_prob)
        fold_thresholds.append(fold_threshold)

        gaussian, bernoulli = fit_hybrid_nb(
            X_train_cont,
            y_train,
            X_train_bin,
            priors
        )

        p_dead = predict_hybrid_nb(
            gaussian,
            bernoulli,
            X_test_cont,
            X_test_bin
        )

        true_labels.append(y_test.values[0])
        prob_dead_nb.append(p_dead[0])
        final_pred.append(int(p_dead[0] >= fold_threshold))

        print(
            f"Patient {len(true_labels)} "
            f"True={y_test.values[0]} "
            f"P(Dead)={p_dead[0]:.3f} "
            f"Threshold={fold_threshold:.2f} "
            f"Pred={final_pred[-1]}"
        )

    print("\n==============================")
    print("HYBRID BAYESIAN MODEL (IMC dropped, nested threshold, class-freq priors)")
    print("==============================")

    print(
        "\nROC-AUC:",
        roc_auc_score(true_labels, prob_dead_nb)
    )

    print(
        "\nMean fold threshold:",
        np.mean(fold_thresholds),
        "std:",
        np.std(fold_thresholds)
    )

    print("\nClassification Report")
    print(classification_report(true_labels, final_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(true_labels, final_pred))


if not true_labels and (RUN_MODELS["rf"] or RUN_MODELS["lr"]):

    true_labels = list(y)


if RUN_MODELS["rf"]:

    for train_idx, test_idx in loo.split(X_processed):

        X_train = X_processed.iloc[train_idx]
        X_test = X_processed.iloc[test_idx]

        y_train = y.iloc[train_idx]

        rf_params = {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 4, 8, 12],
            "min_samples_leaf": [1, 2, 4]
        }

        rf_search = GridSearchCV(
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42
            ),
            rf_params,
            scoring="roc_auc",
            cv=5
        )

        rf_search.fit(X_train, y_train)

        rf = rf_search.best_estimator_

        p_rf = rf.predict_proba(X_test)

        rf_predictions.append(p_rf[0][1])
        rf_preds.append(rf.predict(X_test)[0])

    print("\n==============================")
    print("RANDOM FOREST LOO")
    print("==============================")

    print(
        "ROC-AUC:",
        roc_auc_score(true_labels, rf_predictions)
    )

    print("\nClassification Report")
    print(classification_report(true_labels, rf_preds))

    print("\nConfusion Matrix")
    print(confusion_matrix(true_labels, rf_preds))


if RUN_MODELS["lr"]:

    for train_idx, test_idx in loo.split(X_processed):

        X_train = X_processed.iloc[train_idx]
        X_test = X_processed.iloc[test_idx]

        y_train = y.iloc[train_idx]

        lr_params = {
            "C": np.logspace(-3, 2, 12),
            "penalty": ["l1", "l2"]
        }

        lr_search = GridSearchCV(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear"
            ),
            lr_params,
            scoring="roc_auc",
            cv=5
        )

        lr_search.fit(X_train, y_train)

        lr = lr_search.best_estimator_

        p_lr = lr.predict_proba(X_test)

        lr_predictions.append(p_lr[0][1])
        lr_preds.append(lr.predict(X_test)[0])

    print("\n==============================")
    print("LOGISTIC REGRESSION LOO")
    print("==============================")

    print(
        "ROC-AUC:",
        roc_auc_score(true_labels, lr_predictions)
    )

    print("\nClassification Report")
    print(classification_report(true_labels, lr_preds))

    print("\nConfusion Matrix")
    print(confusion_matrix(true_labels, lr_preds))


comparison_rows = []

if RUN_MODELS["hybrid_nb"]:
    comparison_rows.append(("Hybrid NB", roc_auc_score(true_labels, prob_dead_nb)))

if RUN_MODELS["rf"]:
    comparison_rows.append(("Random Forest", roc_auc_score(true_labels, rf_predictions)))

if RUN_MODELS["lr"]:
    comparison_rows.append(("Logistic Regression", roc_auc_score(true_labels, lr_predictions)))

if comparison_rows:

    print("\n==============================")
    print("MODEL COMPARISON (ROC-AUC)")
    print("==============================")

    print(
        pd.DataFrame(
            comparison_rows,
            columns=["Model", "ROC_AUC"]
        )
    )


results = X.copy()

if RUN_MODELS["hybrid_nb"]:
    results["True_Label"] = true_labels
    results["Bayesian_P_Dead"] = prob_dead_nb
    results["Bayesian_Threshold"] = fold_thresholds
    results["Bayesian_Pred"] = final_pred

if RUN_MODELS["rf"]:
    if "True_Label" not in results.columns:
        results["True_Label"] = true_labels
    results["RF_P_Dead"] = rf_predictions
    results["RF_Pred"] = rf_preds

if RUN_MODELS["lr"]:
    if "True_Label" not in results.columns:
        results["True_Label"] = true_labels
    results["LR_P_Dead"] = lr_predictions
    results["LR_Pred"] = lr_preds