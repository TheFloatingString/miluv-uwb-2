import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from tabpfn import TabPFNClassifier
import termcolor


def is_nlos(tag_id) -> bool:
    tag_id = int(tag_id)
    if tag_id in [1, 3, 4]:
        return True
    return False


def run_experiment(dataset_name: str, features: list[str]):
    print(f"Running experiment with dataset: {dataset_name}")
    print(f"Using features: {features}")

    if dataset_name == "cirObstacles_1_random3_0":
        df = pd.read_csv(
            "data/source_data/miluv/cirObstacles_1_random3_0/ifo001/uwb_passive.csv"
        )

    elif dataset_name == "cirObstaclesOneTag_1_static_0":
        df = pd.read_csv(
            "data/source_data/miluv/cirObstaclesOneTag_1_static_0/ifo001/uwb_passive.csv"
        )


    elif dataset_name=="cirObstacles_3_random_0":
        # read ifo001 to ifo003
        dfs = []
        for i in range(1, 4):
            df_temp = pd.read_csv(f"data/source_data/miluv/cirObstacles_3_random_0/ifo00{i}/uwb_passive.csv")
            dfs.append(df_temp)
        df = pd.concat(dfs, ignore_index=True)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Loaded dataset with {len(df)} rows from multiple ifo files")
    print(df.head())
    print(f"Columns: {list(df.columns)}")
    print(df.describe())
    print("\nData info:")
    print(df.info())

    X_data = []
    y_data = []

    # X_data is the selected features
    X_data = df[features].values
    y_data = np.array([is_nlos(row["to_id"]) for _, row in df.iterrows()])

    # Filter out any rows with NaN values in the selected features
    mask = ~np.isnan(X_data).any(axis=1)
    X_data = X_data[mask]
    y_data = y_data[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training set NLoS ratio: {np.mean(y_train):.3f}")
    print(f"Test set NLoS ratio: {np.mean(y_test):.3f}")

    # Train a simple classifier
    clf = TabPFNClassifier(device='cpu')
    clf.fit(X_train, y_train)

    train_acc_score = clf.score(X_train, y_train)
    test_acc_score = clf.score(X_test, y_test)
    train_f1_score = f1_score(y_train, clf.predict(X_train))
    test_f1_score = f1_score(y_test, clf.predict(X_test))

    print(f"Training accuracy: {train_acc_score:.3f}")
    print(f"Training F1: {train_f1_score:.3f}")

    print(termcolor.colored(f"Test accuracy: {test_acc_score:.3f}", "yellow"))
    print(termcolor.colored(f"Test F1: {test_f1_score:.3f}", "yellow"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="cirObstacles_1_random3_0")
    # add flags for --skew1, --skew2, --skew3
    parser.add_argument("--skew1", action="store_true", help="Use skew1 feature")
    parser.add_argument("--skew2", action="store_true", help="Use skew2 feature")
    parser.add_argument("--skew3", action="store_true", help="Use skew3 feature")
    
    args = parser.parse_args()

    # Build feature list based on flags
    features = []
    if args.skew1:
        features.append("skew1")
    if args.skew2:
        features.append("skew2")
    if args.skew3:
        features.append("skew3")
    
    if not features:
        features = ["skew1", "skew2", "skew3"]  # default to all if none specified
    
    print(f"Using features: {features}")
    
    run_experiment(args.dataset_name, features)
