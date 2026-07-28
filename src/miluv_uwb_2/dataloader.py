import pandas as pd
from sklearn.model_selection import train_test_split
from rich import print as rprint
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DEBUG = True

METADATA_DICT = {
    "miluv_cirObstacles_1_random3_0": {
        "csv_filepaths": [
            "./data/source_data/miluv/cirObstacles_1_random3_0/ifo001/uwb_range.csv"
        ],
        "nlos_to_ids": [1, 3, 4],
        "dataset_family": "miluv",
    },
    "miluv_cirObstacles_3_random_0": {
        "csv_filepaths": [
            "./data/source_data/miluv/cirObstacles_3_random_0/ifo001/uwb_range.csv",
            "./data/source_data/miluv/cirObstacles_3_random_0/ifo002/uwb_range.csv",
            "./data/source_data/miluv/cirObstacles_3_random_0/ifo003/uwb_range.csv",
        ],
        "nlos_to_ids": [1, 3, 4],
        "dataset_family": "miluv",
    },
    "miluv_cirObstaclesOneTag_1_static_0": {
        "csv_filepaths": [
            "./data/source_data/miluv/cirObstaclesOneTag_1_static_0/ifo001/uwb_range.csv"
        ],
        "nlos_to_ids": [1, 3, 4],
        "dataset_family": "miluv",
    },
    "husky_uwb_los_all": {
        "csv_filepaths": ["./data/source_data/husky/calibrated_uwb_los_all.csv"],
        "nlos_to_ids": [],
        "dataset_family": "husky",
    },
    "husky_uwb_nlos_0_1_4": {
        "csv_filepaths": ["./data/source_data/husky/calibrated_uwb_nlos_0_1_4.csv"],
        "nlos_to_ids": [0, 1, 4],
        "dataset_family": "husky",
    },
    "husky_uwb_nlos_2_3_11": {
        "csv_filepaths": ["./data/source_data/husky/calibrated_uwb_nlos_2_3_11.csv"],
        "nlos_to_ids": [2, 3, 11],
        "dataset_family": "husky",
    },
    "husky_uwb_nlos_3_5_11": {
        "csv_filepaths": ["./data/source_data/husky/calibrated_uwb_nlos_3_5_11.csv"],
        "nlos_to_ids": [3, 5, 11],
        "dataset_family": "husky",
    },
    "husky_uwb_nlos_all_nonsevere": {
        "csv_filepaths": [
            "./data/source_data/husky/calibrated_uwb_nlos_all_nonsevere.csv"
        ],
        "nlos_to_ids": [0, 1, 2, 3, 4, 5, 11],
        "dataset_family": "husky",
    },
}


class UwbDataset:
    def __init__(self, dataset_name: str):
        """
        loads single dataset

        dataset_name:
        - miluv_cirObstacles_1_random3_0
        - miluv_cirObstacles_3_random_0
        - miluv_cirObstaclesOneTag_1_static_0
        - husky_uwb_los_all
        - husky_uwb_nlos_0_1_4
        - husky_uwb_nlos_2_3_11
        - husky_uwb_nlos_3_5_11
        - husky_uwb_nlos_all_nonsevere
        """
        # read all csv files for this dataset
        self.df_single_dataset = pd.concat(
            [
                pd.read_csv(csv_filepath)
                for csv_filepath in METADATA_DICT[dataset_name]["csv_filepaths"]
            ]
        )
        self.all_to_ids = self.df_single_dataset["to_id"].unique()
        raw_nlos_ids = METADATA_DICT[dataset_name].get("nlos_to_ids", [])
        self.nlos_to_ids = [id for id in self.all_to_ids if id in raw_nlos_ids]
        self.los_to_ids = [id for id in self.all_to_ids if id not in self.nlos_to_ids]
        # add is_nlos column: 1 if True, 0 if False
        self.df_single_dataset["is_nlos"] = self.df_single_dataset["to_id"].isin(
            self.nlos_to_ids
        ).astype(int)
        # add dataset_family column
        self.dataset_family = METADATA_DICT[dataset_name]["dataset_family"]

    def __getitem__(self, index):
        return self.df_single_dataset.iloc[index]

    def __len__(self):
        return len(self.df_single_dataset)


class UwbSingleDatasetLoader:
    """
    Bundles
    [{"X_train": np.ndarray, "X_test": np.ndarray, "y_train": np.ndarray, "y_test": np.ndarray, "metadata": str}]
    """

    def __init__(
        self,
        uwb_dataset: UwbDataset,
        y_cols: list[str] = ["is_nlos"],
        window_size: int = 10,
    ):
        self.dataset = uwb_dataset
        df = self.dataset.df_single_dataset

        tx1, tx2, tx3 = df["tx1_raw"].values, df["tx2_raw"].values, df["tx3_raw"].values
        rx1, rx2, rx3 = df["rx1_raw"].values, df["rx2_raw"].values, df["rx3_raw"].values
        self.X_data = np.column_stack([
            tx1 - tx2, tx1 - tx3, tx2 - tx3,
            rx1 - rx2, rx1 - rx3, rx2 - rx3,
        ])
        self.y_data = df[y_cols].values
        self.row_to_ids = self.dataset.df_single_dataset["to_id"].values

        # Extract IDs for each dataset
        self.list_of_nlos_ids = self.dataset.nlos_to_ids
        self.list_of_los_ids = self.dataset.los_to_ids
        self.list_of_all_ids = self.dataset.all_to_ids
        self.window_size = window_size

    def _make_windows(self, X: np.ndarray, y: np.ndarray, row_ids: np.ndarray, unique_ids) -> tuple:
        """Window per tag to avoid cross-tag boundary windows, aggregate with mean+std."""
        X_wins, y_wins = [], []
        for tid in unique_ids:
            mask = row_ids == tid
            Xt, yt = X[mask], y[mask]
            for i in range(0, len(Xt) - self.window_size + 1, self.window_size):
                window = Xt[i:i + self.window_size]
                X_wins.append(np.concatenate([np.mean(window, axis=0), np.std(window, axis=0)]))
                y_wins.append(yt[i + self.window_size - 1])
        return np.array(X_wins), np.array(y_wins).ravel()

    def generate_train_test_split(self, mode: str = "random"):
        """
        mode: "random" or "modified_loocv"

        modified_loocv: cross-validation where two tags (one LOS, one NLOS) are kept in test set, and remaining are in train set
        """
        if mode == "random":
            if self.window_size == 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X_data, self.y_data.ravel(), test_size=0.2, random_state=42
                )
                return [
                    {
                        "X_train": X_train,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_test": y_test,
                        "metadata": "random",
                    }
                ]
            else:
                X_windowed, y_windowed = self._make_windows(
                    self.X_data, self.y_data, self.row_to_ids, self.list_of_all_ids
                )
                return [
                    {
                        "X_train": X_windowed,
                        "X_test": X_windowed,
                        "y_train": y_windowed,
                        "y_test": y_windowed,
                        "metadata": "random",
                    }
                ]
        elif mode == "modified_loocv":
            result = []
            list_of_train_test_to_ids = []
            nlos_ids = self.dataset.nlos_to_ids
            los_ids = self.dataset.los_to_ids
            LEN_LOS_IDS = len(los_ids)
            for i in range(len(nlos_ids)):
                test_ids = (nlos_ids[i], los_ids[i % LEN_LOS_IDS])
                train_ids = list(set(self.list_of_all_ids) - set(test_ids))
                list_of_train_test_to_ids.append({"train_ids": train_ids, "test_ids": test_ids})

            if DEBUG:
                rprint(list_of_train_test_to_ids)

            for train_test_to_ids in list_of_train_test_to_ids:
                train_ids = train_test_to_ids["train_ids"]
                test_ids = train_test_to_ids["test_ids"]
                train_mask = np.isin(self.row_to_ids, train_ids)
                test_mask = np.isin(self.row_to_ids, test_ids)

                if self.window_size == 1:
                    result.append(
                        {
                            "X_train": self.X_data[train_mask],
                            "X_test": self.X_data[test_mask],
                            "y_train": self.y_data[train_mask].ravel(),
                            "y_test": self.y_data[test_mask].ravel(),
                            "metadata": f"modified_loocv_{test_ids}",
                        }
                    )
                else:
                    X_train_w, y_train_w = self._make_windows(
                        self.X_data[train_mask], self.y_data[train_mask],
                        self.row_to_ids[train_mask], train_ids
                    )
                    X_test_w, y_test_w = self._make_windows(
                        self.X_data[test_mask], self.y_data[test_mask],
                        self.row_to_ids[test_mask], test_ids
                    )
                    result.append(
                        {
                            "X_train": X_train_w,
                            "X_test": X_test_w,
                            "y_train": y_train_w,
                            "y_test": y_test_w,
                            "metadata": f"modified_loocv_{test_ids}",
                        }
                    )
            return result

        else:
            raise ValueError("mode must be 'random' or 'modified_loocv'")

class UwbCrossDatasetLoader:
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


if __name__ == "__main__":
    dataset = UwbDataset(dataset_name="husky_uwb_nlos_0_1_4")
    print(dataset[0:10])
    print(len(dataset))
    print(dataset.df_single_dataset.head())

    # test dataloader loocv modified
    single_dl = UwbSingleDatasetLoader(dataset, window_size=1)
    result = single_dl.generate_train_test_split("modified_loocv")
    

    scores = []
    for i in range(len(result)):

        clf = RandomForestClassifier()
        clf.fit(result[i]["X_train"], result[i]["y_train"])
        score = clf.score(result[i]["X_test"], result[i]["y_test"])
        scores.append(score)
        print(f"Score: {score}")
        print(f"Number of samples: {len(result[i]['X_train']) + len(result[i]['X_test'])}")
        print(f"Number of windows: {len(result[i]['X_train'])}")

    print(f"mean accuracy: {np.mean(scores)}")
    print(f"std accuracy: {np.std(scores)}")
