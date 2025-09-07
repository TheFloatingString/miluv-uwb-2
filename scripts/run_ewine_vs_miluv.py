from typing import Literal
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from tabpfn import TabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from rich import print as rprint

import wandb

from miluv_uwb_2.utils import is_nlos_miluv

np.random.seed(42)

wandb.login()


LIST_OF_PANDAS_NLOS_SET = [
    "data/source_data/ewine/2016_paper/uwb_dataset_part1.csv",
    "data/source_data/ewine/2016_paper/uwb_dataset_part2.csv",
    "data/source_data/ewine/2016_paper/uwb_dataset_part3.csv",
    "data/source_data/ewine/2016_paper/uwb_dataset_part4.csv",
    "data/source_data/ewine/2016_paper/uwb_dataset_part5.csv",
    "data/source_data/ewine/2016_paper/uwb_dataset_part6.csv",
    "data/source_data/ewine/2016_paper/uwb_dataset_part7.csv",
]

LIST_OF_PANDAS_LOCALIZATION_SET = [
    # "data/source_data/ewine/2018_paper/dataset1_tag_room0.csv", # commenting out due to nan in last col
    # "data/source_data/ewine/2018_paper/dataset1_tag_room1.csv", # commenting out due to nan in last col
    "data/source_data/ewine/2018_paper/dataset2_tag_room0.csv",
    "data/source_data/ewine/2018_paper/dataset2_tag_room1_part0.csv",
    "data/source_data/ewine/2018_paper/dataset2_tag_room1_part1.csv",
    "data/source_data/ewine/2018_paper/dataset2_tag_room1_part2.csv",
    "data/source_data/ewine/2018_paper/dataset2_tag_room1_part3.csv",
    "data/source_data/ewine/2018_paper/dataset2_tag_room1_part4.csv",
    "data/source_data/ewine/2018_paper/dataset2_tag_room1_part5.csv",
    "data/source_data/ewine/2018_paper/dataset2_tag_room1_part6.csv",
    "data/source_data/ewine/2018_paper/dataset2_tag_room1_part7.csv",
    "data/source_data/ewine/2018_paper/dataset2_tag_room1_part8.csv",
    "data/source_data/ewine/2018_paper/dataset2_tag_room1_part9.csv",
]

SOURCE_DATA_VALUES = [
    "EWINE_LOCALIZATION_SET",
    "EWINE_NLOS_SET",
    "MILUV_STATIC_1_UAV",
    "MILUV_RANDOM_1_UAV",
    "MILUV_RANDOM_3_UAV",
]


def main(
    source_data: SOURCE_DATA_VALUES,
    model: Literal["random_forest", "tabpfn"],
    subsample: float,
    last_500_cir_cols_only: bool,
    max_10000_rows: bool,
):
    run = wandb.init(project="miluv-uwb-2", reinit=True)

    if source_data == "EWINE_LOCALIZATION_SET":
        list_of_pd_dfs = []
        for file in LIST_OF_PANDAS_LOCALIZATION_SET:
            ewine_df = pd.read_csv(file, header=None)
            print(ewine_df.head())
            list_of_pd_dfs.append(ewine_df)

        ewine_df = pd.concat(list_of_pd_dfs)

        if max_10000_rows:
            ewine_df = ewine_df.sample(n=10000, random_state=42)
        elif 0 < subsample < 1:
            ewine_df = ewine_df.sample(frac=subsample, random_state=42)

        # get last 1,016 columns as X_data
        if last_500_cir_cols_only:
            X_data = ewine_df.iloc[:, -500:]
        else:
            X_data = ewine_df.iloc[:, -1016:]
        print(X_data.head())
        print(X_data.shape)

        y_data = ewine_df.iloc[:, 5].apply(lambda x: int(x))
        print(y_data.head())
        print(y_data.unique())
        print(y_data.value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

    elif source_data == "EWINE_NLOS_SET":
        list_of_pd_dfs = []

        for file in LIST_OF_PANDAS_NLOS_SET:
            curr_ewine_df = pd.read_csv(file)
            list_of_pd_dfs.append(curr_ewine_df)

        ewine_df = pd.concat(list_of_pd_dfs)
        if last_500_cir_cols_only:
            ewine_df = ewine_df.iloc[:, -500:]
        else:
            ewine_df = ewine_df.iloc[:, -1016:]

        # print(ewine_df.iloc[:,0:15].head())
        print(ewine_df.iloc[:, 15:].head())
        print(ewine_df.iloc[:, 0].head())

        # subsample
        if max_10000_rows:
            ewine_df = ewine_df.sample(n=10000, random_state=42)
        elif 0 < subsample < 1:
            ewine_df = ewine_df.sample(frac=subsample, random_state=42)

        range_dist = ewine_df.iloc[:, 1].values
        y_data = ewine_df.iloc[:, 0].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

    elif source_data == "MILUV_STATIC_1_UAV":
        miluv_df = pd.read_csv(
            "data/source_data/miluv/cirObstaclesOneTag_1_static_0/ifo001/uwb_cir.csv"
        )

        if max_10000_rows:
            miluv_df = miluv_df.sample(n=10000, random_state=42)
        elif 0 < subsample < 1:
            miluv_df = miluv_df.sample(frac=subsample, random_state=42)

        X_data = np.array([eval(x) for x in miluv_df["cir"].values])
        if last_500_cir_cols_only:
            X_data = X_data[:, -500:]
        else:
            X_data = X_data[:, -1016:]
        y_data = miluv_df["to_id"].apply(is_nlos_miluv).values

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )

        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

    elif source_data == "MILUV_RANDOM_1_UAV":
        miluv_df = pd.read_csv(
            "data/source_data/miluv/cirObstacles_1_random3_0/ifo001/uwb_cir.csv"
        )

        if max_10000_rows:
            miluv_df = miluv_df.sample(n=10000, random_state=42)
        elif 0 < subsample < 1:
            miluv_df = miluv_df.sample(frac=subsample, random_state=42)

        X_data = np.array([eval(x) for x in miluv_df["cir"].values])
        if last_500_cir_cols_only:
            X_data = X_data[:, -500:]
        else:
            X_data = X_data[:, -1016:]
        y_data = miluv_df["to_id"].apply(is_nlos_miluv).values

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )

    elif source_data == "MILUV_RANDOM_3_UAV":
        miluv_df1 = pd.read_csv(
            "data/source_data/miluv/cirObstacles_3_random_0/ifo001/uwb_cir.csv"
        )
        miluv_df2 = pd.read_csv(
            "data/source_data/miluv/cirObstacles_3_random_0/ifo002/uwb_cir.csv"
        )
        miluv_df3 = pd.read_csv(
            "data/source_data/miluv/cirObstacles_3_random_0/ifo003/uwb_cir.csv"
        )
        miluv_df = pd.concat([miluv_df1, miluv_df2, miluv_df3])
        # drop rows in miluv_df where to_id is larger than 6
        # miluv_df = miluv_df[miluv_df["to_id"] <= 6]
        print(miluv_df.head())
        if 0 < subsample < 1:
            miluv_df = miluv_df.sample(frac=subsample, random_state=42)

        X_data = np.array([eval(x) for x in miluv_df["cir"].values])
        if last_500_cir_cols_only:
            X_data = X_data[:, -500:]
        else:
            X_data = X_data[:, -1016:]
        if max_10000_rows:
            X_data = X_data[:10000]
        y_data = miluv_df["to_id"].apply(is_nlos_miluv).values

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )

        print(subsample)

        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

    # clf = TabPFNClassifier(ignore_pretraining_limits=True)
    if model == "random_forest":
        clf = RandomForestClassifier()
    elif model == "tabpfn":
        clf = TabPFNClassifier()
    elif model == "svc":
        clf = SVC()
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
        default="EWINE_NLOS_SET",
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
    print("added args")
    args = parser.parse_args()
    main(
        args.source_data,
        args.model,
        args.subsample,
        args.last_500_cir_cols_only,
        args.max_10000_rows,
    )
