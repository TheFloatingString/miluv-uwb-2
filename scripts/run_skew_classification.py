from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier
from glob import glob
import pandas as pd
import numpy as np
import argparse
from rich import print as rprint
from sklearn.metrics import f1_score


import warnings

warnings.filterwarnings("ignore")


SAME_TIME_EVALS = [
    {"name": "random-1", "files": ["miluv-random_1-ifo001-uwb_range"]},
    {"name": "static-1", "files": ["miluv-static_1-ifo001-uwb_range"]},
    {
        "name": "random-3",
        "files": [
            "miluv-random_3-ifo001-uwb_range",
            "miluv-random_3-ifo002-uwb_range",
            "miluv-random_3-ifo003-uwb_range",
        ],
    },
]


ABLATIONS = [
    {"name": "fpp1 baseline", "features": ["fpp1"]},
    {"name": "skew 1 only", "features": ["skew1"]},
    {"name": "skew 2 only", "features": ["skew2"]},
    {"name": "skew 1 and 2", "features": ["skew1", "skew2"]},
    {"name": "skew 1 and 2 and range", "features": ["skew1", "skew2", "range_raw"]},
    {
        "name": "skew 1 and 2 and range and bias",
        "features": ["skew1", "skew2", "range_raw", "bias_raw"],
    },
    {
        "name": "skew 1 and 2 and range and bias and tx/rx",
        "features": [
            "skew1",
            "skew2",
            "range_raw",
            "bias_raw",
            "tx1",
            "tx2",
            "tx3",
            "rx1",
            "rx2",
            "rx3",
        ],
    },
]


def is_nlos(tag_id) -> int:
    # NLOS
    if tag_id in [1, 3, 4]:
        return 1
    # LOS
    else:
        return 0


def run_single_ablation(ablation, classifier_choice):
    for evals in SAME_TIME_EVALS:
        list_of_files = []
        for eval_root in evals["files"]:
            list_of_files.extend(glob(f"data/processed_data/{eval_root}_*.csv"))
        # by convention, the first 4 are train, the last 4 are test
        train_files = list_of_files[:4]
        test_files = list_of_files[4:]
        df_train = pd.concat([pd.read_csv(f) for f in train_files])
        df_test = pd.concat([pd.read_csv(f) for f in test_files])

        X_train = df_train[ablation["features"]].values
        X_test = df_test[ablation["features"]].values

        y_train = df_train["to_id"].apply(is_nlos).values
        y_test = df_test["to_id"].apply(is_nlos).values

        if classifier_choice == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_choice == "tabpfn":
            clf = TabPFNClassifier()
        else:
            raise ValueError(f"Unknown classifier choice: {classifier_choice}")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        correct_prds = y_pred == y_test

        acc = np.mean(correct_prds)
        stderr = np.std(correct_prds) / np.sqrt(len(correct_prds))
        rprint(f"Test case: {evals['name']}")
        rprint(f"Accuracy: {round(acc, 3)} +/- {round(stderr, 3)}")

        f1 = f1_score(y_test, y_pred)
        rprint(f"F1 Score: {round(f1, 3)}")

        rprint(f"list of ablations: {ablation['features']}")
        rprint("-" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="random_forest")
    args = parser.parse_args()

    for ablation in ABLATIONS:
        run_single_ablation(ablation, classifier_choice=args.model)
