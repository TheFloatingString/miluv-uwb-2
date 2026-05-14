import ast
from typing import Literal
import argparse
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from rich import print as rprint
import glob

np.random.seed(42)

SOURCE_DATA_VALUES = ["MILUV_STATIC_1_UAV", "MILUV_RANDOM_1_UAV", "MILUV_RANDOM_3_UAV"]


def main(
    case: Literal["MILUV_STATIC_1_UAV", "MILUV_RANDOM_1_UAV", "MILUV_RANDOM_3_UAV"],
    model: Literal["svc"],
    subsample: float,
    last_500_cir_cols_only: bool,
    max_10000_rows: bool,
    ablations: list[str],
):
    if case == "MILUV_STATIC_1_UAV":
        prefix = "static"
        df_cir_0 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_0.csv"
        )
        df_cir_1 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_1.csv"
        )
        df_cir_2 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_2.csv"
        )
        df_cir_3 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_3.csv"
        )
        df_cir_4 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_4.csv"
        )
        df_cir_5 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_5.csv"
        )

        df_range_0 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_range_0.csv"
        )
        df_range_1 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_range_1.csv"
        )
        df_range_2 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_range_2.csv"
        )
        df_range_3 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_range_3.csv"
        )
        df_range_4 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_range_4.csv"
        )
        df_range_5 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_range_5.csv"
        )

        df_cir_miluv = pd.concat(
            [df_cir_0, df_cir_1, df_cir_2, df_cir_3, df_cir_4, df_cir_5]
        )
        # drop rows where from_id != 10
        df_cir_miluv = df_cir_miluv[df_cir_miluv["from_id"] == 10]

        df_range_miluv = pd.concat(
            [df_range_0, df_range_1, df_range_2, df_range_3, df_range_4, df_range_5]
        )

    elif case == "MILUV_RANDOM_1_UAV":
        prefix = "random"
        df_cir_0 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_0.csv"
        )
        df_cir_1 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_1.csv"
        )
        df_cir_2 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_2.csv"
        )
        df_cir_3 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_3.csv"
        )
        df_cir_4 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_4.csv"
        )
        df_cir_5 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_5.csv"
        )

        df_range_0 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_range_0.csv"
        )
        df_range_1 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_range_1.csv"
        )
        df_range_2 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_range_2.csv"
        )
        df_range_3 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_range_3.csv"
        )
        df_range_4 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_range_4.csv"
        )
        df_range_5 = pd.read_csv(
            f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_range_5.csv"
        )

        df_cir_miluv = pd.concat(
            [df_cir_0, df_cir_1, df_cir_2, df_cir_3, df_cir_4, df_cir_5]
        )
        df_range_miluv = pd.concat(
            [df_range_0, df_range_1, df_range_2, df_range_3, df_range_4, df_range_5]
        )

    elif case == "MILUV_RANDOM_3_UAV":
        prefix = "random"

        list_of_cir_filepaths = glob.glob(
            f"data/processed_data/miluv-{prefix}_3-ifo00*-uwb_cir_*.csv"
        )
        list_of_cir_dfs = [pd.read_csv(filepath) for filepath in list_of_cir_filepaths]

        list_of_ranging_filepaths = glob.glob(
            f"data/processed_data/miluv-{prefix}_3-ifo00*-uwb_range_*.csv"
        )
        list_of_ranging_dfs = [
            pd.read_csv(filepath) for filepath in list_of_ranging_filepaths
        ]

        df_cir_miluv = pd.concat(list_of_cir_dfs)
        df_range_miluv = pd.concat(list_of_ranging_dfs)

        print(f"df_cir_miluv shape: {df_cir_miluv.shape}")
        print(f"df_range_miluv shape: {df_range_miluv.shape}")

    def is_nlos_miluv(tag_id):
        if tag_id in [1, 3, 4]:
            return True
        return False

    if max_10000_rows:
        df_cir_miluv = df_cir_miluv.sample(n=10000, random_state=42)
    elif 0 < subsample < 1:
        df_cir_miluv = df_cir_miluv.sample(frac=subsample, random_state=42)

    X_data = []
    y_data = []

    for i in tqdm.trange(len(df_cir_miluv)):
        cir = np.asarray(eval(df_cir_miluv.iloc[i]["cir"]), dtype=np.float64)

        if last_500_cir_cols_only:
            cir = cir[-500:]

        # get the closest range column value based on timestamp
        timestamp = df_cir_miluv.iloc[i]["timestamp"]
        # make sure df_range from_id and to_id cols match df_cir from_id and to_id cols

        # define a new dataframe df_range_curr that is filtered by to_id and from_id
        tmp_df = df_range_miluv[
            df_range_miluv["to_id"] == df_cir_miluv.iloc[i]["to_id"]
        ]
        tmp_df = tmp_df[tmp_df["from_id"] == df_cir_miluv.iloc[i]["from_id"]]

        range_idx = (tmp_df["timestamp"] - timestamp).abs().idxmin()
        if "ranging_scaling" in ablations:
            range = tmp_df.loc[range_idx, "range"]
            if len(range.shape) > 0:
                range = range.values[0]
            X_data.append((range**2) * cir)
        elif "distance_scaling" in ablations:
            range = tmp_df.loc[range_idx, "gt_range"]
            # print(range.shape)
            # print(len(range.shape))
            if len(range.shape) > 0:
                range = range.values[0]
                # print(range)
            # range = range[0]
            # print(tmp_df.loc[range_idx])
            # print(f"range:  {range}")
            X_data.append((range**2) * cir)
        else:
            X_data.append(cir)

        y_data.append(is_nlos_miluv(df_cir_miluv.iloc[i]["to_id"]))

    # move all X_data and y_data here
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)

    # add unified preprocessing (fft, distance scaling, min-max scaling)
    print(f"ablations: {ablations}")
    for ablation in ablations:
        if ablation == "fft":
            X_data = np.real(np.fft.fft(X_data, axis=1))
        elif ablation == "min_max_scaling":
            from sklearn.preprocessing import MinMaxScaler

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
    if model == "svc":
        clf = SVC()
        print("running svc")
    elif model == "tabpfn":
        from tabpfn import TabPFNClassifier

        clf = TabPFNClassifier()
        print("running tabPFN")
    elif model == "random_forest":
        clf = RandomForestClassifier()
        print("running random_forest")
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
    rprint(f"on case {case}")
    rprint(f"with model {model}")
    rprint(f"with subsample {subsample}")
    rprint(f"with ablations {ablations}")

    rprint(f"X_data.shape: {X_data.shape}")
    rprint(f"NLOS pct: {round(sum(y_data) / len(y_data), 3)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="svc",
    )
    parser.add_argument(
        "--case",
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
        args.case,
        args.model,
        args.subsample,
        args.last_500_cir_cols_only,
        args.max_10000_rows,
        ablations_list,
    )
