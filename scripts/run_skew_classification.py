from sklearn.ensemble import RandomForestClassifier
# from tabpfn import TabPFNClassifier
from glob import glob
import pandas as pd
import numpy as np
import argparse
from rich import print as rprint
from sklearn.metrics import f1_score
from miluv_uwb_2.utils import is_nlos_miluv, get_obstacle_type_miluv
import wandb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import warnings

np.random.seed(42)

wandb.login()

# warnings.filterwarnings("ignore")


SAME_TIME_EVALS = [
    {"dataset_name": "random-1", "files": ["miluv-random_1-ifo001-uwb_range"]},
    {"dataset_name": "static-1", "files": ["miluv-static_1-ifo001-uwb_range"]},
    {
        "dataset_name": "random-3",
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
    {
        "name": "skew 1 and 2 and range and bias",
        "features": ["skew1", "skew2", "range"],
    },
    {
        "name": "skew 1 and 2 and range and bias and tx/rx",
        "features": [
            "skew1",
            "skew2",
            "range",
            "tx1",
            "tx2",
            "tx3",
            "rx1",
            "rx2",
            "rx3",
        ],
    },
]


def run_single_ablation(ablation, classifier_choice, train_test_split_type, task):
    for evals in SAME_TIME_EVALS:
        run = wandb.init(project="miluv-uwb-2-skew-exp", reinit=True)
        list_of_files = []
        for eval_root in evals["files"]:
            list_of_files.extend(glob(f"data/processed_data/{eval_root}_*.csv"))
        if train_test_split_type == "4-vs-2":
            all_y_tests = []
            all_y_preds = []
            all_y_correct_prds = []

            for i in range(3):
                if i == 0:
                    train_files = list_of_files[:4]
                    test_files = list_of_files[4:]
                elif i == 1:
                    train_files = list_of_files[:2] + list_of_files[4:]
                    test_files = list_of_files[2:4]
                elif i == 2:
                    train_files = list_of_files[2:]
                    test_files = list_of_files[:2]
                df_train = pd.concat([pd.read_csv(f) for f in train_files])
                df_test = pd.concat([pd.read_csv(f) for f in test_files])

                X_train = df_train[ablation["features"]].values
                X_test = df_test[ablation["features"]].values

                if task == "nlos":
                    y_train = df_train["to_id"].apply(is_nlos_miluv).values
                    y_test = df_test["to_id"].apply(is_nlos_miluv).values
                elif task == "obstacle_type":
                    y_train = df_train["to_id"].apply(get_obstacle_type_miluv).values
                    y_test = df_test["to_id"].apply(get_obstacle_type_miluv).values

                if classifier_choice == "random_forest":
                    clf = RandomForestClassifier()
                elif classifier_choice == "tabpfn":
                    clf = TabPFNClassifier()
                elif classifier_choice == "svc":
                    clf = SVC()
                else:
                    raise ValueError(f"Unknown classifier choice: {classifier_choice}")
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                correct_prds = y_pred == y_test

                all_y_tests.extend(y_test)
                all_y_preds.extend(y_pred)
                all_y_correct_prds.extend(correct_prds)

                print(f"ran cross-val {i}")

            acc = np.mean(all_y_correct_prds)
            acc_stderr = np.std(all_y_correct_prds) / np.sqrt(len(all_y_correct_prds))
            rprint(f"Test case: {evals['dataset_name']}")
            rprint(f"Accuracy: {round(acc, 3)} +/- {round(acc_stderr, 3)}")

            f1 = f1_score(all_y_tests, all_y_preds, average="macro")
            f1_stderr = np.std(all_y_correct_prds) / np.sqrt(len(all_y_correct_prds))
            rprint(f"F1 Score: {round(f1, 3)} +/- {round(f1_stderr, 3)}")

            rprint(f"list of ablations: {ablation['features']}")
            rprint("-" * 20)

        elif train_test_split_type == "80-20":
            df_miluv = pd.concat([pd.read_csv(f) for f in list_of_files])

            if task == "nlos":
                y_data = df_miluv["to_id"].apply(is_nlos_miluv).values
            elif task == "obstacle_type":
                y_data = df_miluv["to_id"].apply(get_obstacle_type_miluv).values

            X_train, X_test, y_train, y_test = train_test_split(
                df_miluv[ablation["features"]].values,
                y_data,
                test_size=0.2,
                random_state=42,
            )

            if classifier_choice == "random_forest":
                clf = RandomForestClassifier()
            elif classifier_choice == "tabpfn":
                from tabpfn import TabPFNClassifier
                clf = TabPFNClassifier()
            elif classifier_choice == "svc":
                clf = SVC()
            else:
                raise ValueError(f"Unknown classifier choice: {classifier_choice}")

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = np.mean(y_pred == y_test)
            acc_stderr = np.std(y_pred == y_test) / np.sqrt(len(y_pred == y_test))
            f1 = f1_score(y_test, y_pred, average="macro")
            f1_stderr = np.std(y_pred == y_test) / np.sqrt(len(y_pred == y_test))

            rprint(f"Test case: {evals['dataset_name']}")
            rprint(f"Accuracy: {round(acc, 3)} +/- {round(acc_stderr, 3)}")
            rprint(f"F1 Score: {round(f1, 3)} +/- {round(f1_stderr, 3)}")
            rprint(f"list of ablations: {ablation['features']}")
            rprint("-" * 20)

        wandb.log(
            {
                "accuracy": acc,
                "accuracy_stderr": acc_stderr,
                "f1": f1,
                "f1_stderr": f1_stderr,
                "ablation": ablation["features"],
                "dataset": evals["dataset_name"],
                "task": task,
                "train_test_split_type": train_test_split_type,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="random_forest")
    parser.add_argument(
        "--train-test-split-type",
        type=str,
        default="4-vs-2",
        choices=["4-vs-2", "80-20"],
    )
    parser.add_argument(
        "--task", type=str, default="nlos", choices=["nlos", "obstacle_type"]
    )
    args = parser.parse_args()

    for ablation in ABLATIONS:
        run_single_ablation(
            ablation,
            classifier_choice=args.model,
            train_test_split_type=args.train_test_split_type,
            task=args.task,
        )
