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
    ablations: list[str],
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

        if "distance_scaling" in ablations:
            x_1 = ewine_df[0].to_numpy()
            x_2 = ewine_df[2].to_numpy()
            y_1 = ewine_df[1].to_numpy()
            y_2 = ewine_df[3].to_numpy()
            dist = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
            ewine_df.iloc[:, -1016:] *= dist[:, None] ** 2
        elif "ranging_scaling" in ablations:
            ewine_df.iloc[:, -1016:] *= ewine_df.iloc[:, 4].to_numpy()[:, None] ** 2

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

    elif source_data == "EWINE_NLOS_SET":
        list_of_pd_dfs = []

        for file in LIST_OF_PANDAS_NLOS_SET:
            curr_ewine_df = pd.read_csv(file)
            list_of_pd_dfs.append(curr_ewine_df)

        ewine_df = pd.concat(list_of_pd_dfs)

        # subsample
        if max_10000_rows:
            ewine_df = ewine_df.sample(n=10000, random_state=42)
        elif 0 < subsample < 1:
            ewine_df = ewine_df.sample(frac=subsample, random_state=42)

        if last_500_cir_cols_only:
            X_data = ewine_df.iloc[:, -500:]
        else:
            X_data = ewine_df.iloc[:, -1016:]

        # print(ewine_df.iloc[:,0:15].head())
        print(ewine_df.iloc[:, 15:].head())
        print(ewine_df.iloc[:, 0].head())

        range_dist = ewine_df.iloc[:, 1].values
        y_data = ewine_df.iloc[:, 0].values

        if "distance_scaling" in ablations:
            raise ValueError("Distance scaling not implemented for EWINE NLOS set")
        elif "ranging_scaling" in ablations:
            X_data *= range_dist[:, None] ** 2

    # move all X_data and y_data here
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)

    # add unified preprocessing (fft, distance scaling, min-max scaling)
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
