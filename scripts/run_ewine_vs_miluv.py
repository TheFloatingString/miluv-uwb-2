import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier

SOURCE_DATA = "EWINE_UWB_DATASET_PARTS"

def main():
    if SOURCE_DATA == "EWINE_TAG_ROOM":
        ewine_df = pd.read_csv("data/source_data/ewine/tag_room0.csv", header=None)
        print(ewine_df.head())

        # get last 1,016 columns as X_data
        X_data = ewine_df.iloc[:, 24:-1]
        print(X_data.head())
        print(X_data.shape)

        y_data = ewine_df.iloc[:,5]
        print(y_data.head())
        print(y_data.unique())

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

    elif SOURCE_DATA == "EWINE_UWB_DATASET_PARTS":
        ewine_df = pd.read_csv("data/source_data/ewine/uwb_dataset_part1.csv")
        print(ewine_df.iloc[:, 15:].head())
        print(ewine_df.iloc[:, 0].head())

        X_data = ewine_df.iloc[:, 15:].values
        X_data = X_data[0:500,-500:]
        y_data = ewine_df.iloc[0:500, 0].values

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

    clf = TabPFNClassifier(ignore_pretraining_limits=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="random_forest")
    args = parser.parse_args()
    main()
