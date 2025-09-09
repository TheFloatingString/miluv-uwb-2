import ast
from typing import Literal
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import tqdm

from tabpfn import TabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from rich import print as rprint

import wandb

from miluv_uwb_2.utils import is_nlos_miluv, get_gt_dist_drone_to_anchor

np.random.seed(42)

wandb.login()

SOURCE_DATA_VALUES = [
    "MILUV_STATIC_1_UAV",
    "MILUV_RANDOM_1_UAV",
    "MILUV_RANDOM_3_UAV",
]


def main(
    source_data: SOURCE_DATA_VALUES,
    model: Literal["random_forest", "tabpfn", "svc"],
    subsample: float,
    last_500_cir_cols_only: bool,
    max_10000_rows: bool,
    ablations: list[str],
):
    run = wandb.init(project="miluv-uwb-2", reinit=True)

    if source_data == "MILUV_STATIC_1_UAV":
        miluv_cir_df = pd.read_csv(
            "data/source_data/miluv/cirObstaclesOneTag_1_static_0/ifo001/uwb_cir.csv"
        )
        miluv_range_df = pd.read_csv(
            "data/source_data/miluv/cirObstaclesOneTag_1_static_0/ifo001/uwb_range.csv"
        )

        miluv_df = pd.merge_asof(
            miluv_cir_df,
            miluv_range_df[["to_id", "from_id", "range", "gt_range", "timestamp"]],
            on="timestamp",
            direction="nearest",
            by=["to_id", "from_id"],
        ).dropna()

        print(miluv_df.head())
        print(miluv_df.shape)

        if max_10000_rows:
            miluv_df = miluv_df.sample(n=10000, random_state=42)
        elif 0 < subsample < 1:
            miluv_df = miluv_df.sample(frac=subsample, random_state=42)

        X_data = np.array([eval(x) for x in miluv_df["cir"].values]).astype(np.float64)

        if "distance_scaling" in ablations:
            X_data *= miluv_df["gt_range"].to_numpy()[:, None]
        elif "ranging_scaling" in ablations:
            X_data *= miluv_df["range"].to_numpy()[:, None]
        elif "mocap_scaling" in ablations:
            raise NotImplementedError("mocap_scaling not yet implemented")

        if last_500_cir_cols_only:
            X_data = X_data[:, -500:]
        else:
            X_data = X_data[:, -1016:]

        y_data = miluv_df["to_id"].apply(is_nlos_miluv).values

    elif source_data == "MILUV_RANDOM_1_UAV":
        miluv_cir_df = pd.read_csv(
            "data/source_data/miluv/cirObstacles_1_random3_0/ifo001/uwb_cir.csv"
        )

        miluv_range_df = pd.read_csv(
            "data/source_data/miluv/cirObstacles_1_random3_0/ifo001/uwb_range.csv"
        )

        miluv_df = pd.merge_asof(
            miluv_cir_df,
            miluv_range_df[["to_id", "from_id", "range", "gt_range", "timestamp"]],
            on="timestamp",
            direction="nearest",
            by=["to_id", "from_id"],
        ).dropna()

        print(miluv_df.head())
        print(miluv_df.shape)

        if max_10000_rows:
            miluv_df = miluv_df.sample(n=10000, random_state=42)
        elif 0 < subsample < 1:
            miluv_df = miluv_df.sample(frac=subsample, random_state=42)

        X_data = np.array([eval(x) for x in miluv_df["cir"].values]).astype(np.float64)

        if "distance_scaling" in ablations:
            X_data *= miluv_df["gt_range"].to_numpy()[:, None]
        elif "ranging_scaling" in ablations:
            X_data *= miluv_df["range"].to_numpy()[:, None]
        elif "mocap_scaling" in ablations:
            raise NotImplementedError("mocap_scaling not yet implemented")

        if last_500_cir_cols_only:
            X_data = X_data[:, -500:]
        else:
            X_data = X_data[:, -1016:]
        y_data = miluv_df["to_id"].apply(is_nlos_miluv).values

    elif source_data == "MILUV_RANDOM_3_UAV":
        miluv_cir_df1 = pd.read_csv(
            "data/source_data/miluv/cirObstacles_3_random_0/ifo001/uwb_cir.csv"
        )
        miluv_range_df1 = pd.read_csv(
            "data/source_data/miluv/cirObstacles_3_random_0/ifo001/uwb_range.csv"
        )
        miluv_cir_df2 = pd.read_csv(
            "data/source_data/miluv/cirObstacles_3_random_0/ifo002/uwb_cir.csv"
        )
        miluv_range_df2 = pd.read_csv(
            "data/source_data/miluv/cirObstacles_3_random_0/ifo002/uwb_range.csv"
        )
        miluv_cir_df3 = pd.read_csv(
            "data/source_data/miluv/cirObstacles_3_random_0/ifo003/uwb_cir.csv"
        )
        miluv_range_df3 = pd.read_csv(
            "data/source_data/miluv/cirObstacles_3_random_0/ifo003/uwb_range.csv"
        )

        miluv_df1 = pd.merge_asof(
            miluv_cir_df1,
            miluv_range_df1[["to_id", "from_id", "range", "gt_range", "timestamp"]],
            on="timestamp",
            direction="nearest",
            by=["to_id", "from_id"],
        )

        miluv_df2 = pd.merge_asof(
            miluv_cir_df2,
            miluv_range_df2[["to_id", "from_id", "range", "gt_range", "timestamp"]],
            on="timestamp",
            direction="nearest",
            by=["to_id", "from_id"],
        )

        miluv_df3 = pd.merge_asof(
            miluv_cir_df3,
            miluv_range_df3[["to_id", "from_id", "range", "gt_range", "timestamp"]],
            on="timestamp",
            direction="nearest",
            by=["to_id", "from_id"],
        )

        miluv_df = pd.concat([miluv_df1, miluv_df2, miluv_df3]).dropna()

        print(miluv_df.head())
        print(miluv_df.shape)

        if max_10000_rows:
            miluv_df = miluv_df.sample(n=10000, random_state=42)
        elif 0 < subsample < 1:
            miluv_df = miluv_df.sample(frac=subsample, random_state=42)

        X_data = np.array([eval(x) for x in miluv_df["cir"].values]).astype(np.float64)

        if "distance_scaling" in ablations:
            X_data *= miluv_df["gt_range"].to_numpy()[:, None]
        elif "ranging_scaling" in ablations:
            X_data *= miluv_df["range"].to_numpy()[:, None]
        elif "mocap_scaling" in ablations:
            raise NotImplementedError("mocap_scaling not yet implemented")

        if last_500_cir_cols_only:
            X_data = X_data[:, -500:]
        else:
            X_data = X_data[:, -1016:]

        y_data = miluv_df["to_id"].apply(is_nlos_miluv).values

    # move all X_data and y_data here
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)

    # add unified preprocessing (fft, distance scaling, min-max scaling)
    print(f"ablations: {ablations}")
    for ablation in ablations:
        if ablation == "fft":
            X_data = np.real(np.fft.fft(X_data, axis=1))
        elif ablation == "min_max_scaling":
            X_data = MinMaxScaler().fit_transform(X_data)
        elif ablation == "distance_scaling":
            pass
        elif ablation == "ranging_scaling":
            pass
        elif ablation == "mocap_scaling":
            pass
        else:
            raise ValueError(f"Unknown ablation: {ablation}")

    # train-test-split here
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    rprint(f"sample X_train[0]: {X_train[0]}")

    # clf = TabPFNClassifier(ignore_pretraining_limits=True)
    if model == "random_forest":
        clf = RandomForestClassifier()
        print("running random_forest")
    elif model == "tabpfn":
        clf = TabPFNClassifier()
        print("running tabpfn")
    elif model == "svc":
        clf = SVC()
        print("running svc")
    else:
        raise ValueError(f"Unknown model: {model}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # get stderr of accuracy
    acc_stderr = np.std(y_pred == y_test) / np.sqrt(len(y_pred))
    rprint(f"Accuracy: {round(acc, 3)} +/- {round(acc_stderr, 3)}")

    # implement f1_score
    f1 = f1_score(y_test, y_pred)
    # get stderr of f1
    f1_stderr = np.std(f1_score(y_test, y_pred)) / np.sqrt(len(y_pred))
    rprint(f"F1 Score: {round(f1, 3)} +/- {round(f1_stderr, 10)}")

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    wandb.log(
        {
            "accuracy": acc,
            "accuracy_stderr": acc_stderr,
            "f1": f1,
            "f1_stderr": f1_stderr,
            "model": model,
            "subsample": subsample,
            "source_data": source_data,
            "last_500_cir_cols_only": last_500_cir_cols_only,
            "max_10000_rows": max_10000_rows,
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "y_train_shape": y_train.shape,
            "y_test_shape": y_test.shape,
            "ablations": ablations,
            "to_id_value_counts": str(miluv_df["to_id"].value_counts().to_dict())
            if "MILUV" in source_data
            else None,
        }
    )
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
    )
    parser.add_argument(
        "--source_data",
        type=str,
        default="MILUV_STATIC_1_UAV",
        choices=SOURCE_DATA_VALUES,
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--last_500_cir_cols_only",
        action="store_true",
    )
    parser.add_argument(
        "--max_10000_rows",
        action="store_true",
    )
    parser.add_argument(
        "--ablations",
        type=str,
        default="[]",
    )
    print("added args")
    args = parser.parse_args()
    ablations_list: list = ast.literal_eval(args.ablations)
    main(
        args.source_data,
        args.model,
        args.subsample,
        args.last_500_cir_cols_only,
        args.max_10000_rows,
        ablations_list,
    )
