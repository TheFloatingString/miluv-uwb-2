import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from miluv_uwb_2.utils import is_nlos_miluv
import joblib

import numpy as np


np.random.seed(42)


def main(args):
    df = pd.read_csv(args.data_path)
    print(df.head())

    # X_data = df[["range", "tx1", "tx2","tx3","rx1", "rx2","rx3"]]
    X_data = df[["range", "fpp1", "fpp2"]]
    y_data = df[["range", "skew1", "skew2"]]

    is_nlos = df["to_id"].apply(is_nlos_miluv)

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )
    _, _, y_train_nlos, y_test_nlos = train_test_split(
        X_data, is_nlos, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    # model = KNeighborsRegressor()
    # model = DummyRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, "skew_prediction_model.pkl")
    # model.save("skew_prediction_model.joblib")

    y_pred = model.predict(X_test)

    print("MSE:", mean_squared_error(y_test, y_pred))
    # print("R^2 score:", r2_score(y_test, y_pred))

    X_train, X_test, y_train, y_test = train_test_split(
        y_pred, y_test_nlos, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("F1:", f1_score(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/source_data/miluv/cirObstacles_1_random3_0/ifo001/uwb_range.csv",
    )
    args = parser.parse_args()
    main(args)
