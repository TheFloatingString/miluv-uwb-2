import pandas as pd
from sklearn.svm import SVC
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from typing import Literal

# load miluv-random_1-ifo001-uwb_cir_0.

case: Literal["static_1", "random_1"] = "random_1"

if case == "static_1":
    prefix = "static"
    df_cir_0 = pd.read_csv(f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_0.csv")
    df_cir_1 = pd.read_csv(f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_1.csv")
    df_cir_2 = pd.read_csv(f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_2.csv")
    df_cir_3 = pd.read_csv(f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_3.csv")
    df_cir_4 = pd.read_csv(f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_4.csv")
    df_cir_5 = pd.read_csv(f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_5.csv")

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

if case == "random_1":
    prefix = "random"
    df_cir_0 = pd.read_csv(f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_0.csv")
    df_cir_1 = pd.read_csv(f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_1.csv")
    df_cir_2 = pd.read_csv(f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_2.csv")
    df_cir_3 = pd.read_csv(f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_3.csv")
    df_cir_4 = pd.read_csv(f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_4.csv")
    df_cir_5 = pd.read_csv(f"data/processed_data/miluv-{prefix}_1-ifo001-uwb_cir_5.csv")

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


X_data = []
y_data = []


def is_nlos_miluv(tag_id):
    if tag_id in [1, 3, 4]:
        return True
    return False


for df_cir, df_range in zip(
    [df_cir_0, df_cir_1, df_cir_2, df_cir_3, df_cir_4, df_cir_5],
    [df_range_0, df_range_1, df_range_2, df_range_3, df_range_4, df_range_5],
):
    for i in tqdm.trange(len(df_cir)):
        cir = np.asarray(eval(df_cir.iloc[i]["cir"]), dtype=np.float64)
        # get the closest range column value based on timestamp
        timestamp = df_cir.iloc[i]["timestamp"]
        # make sure df_range from_id and to_id cols match df_cir from_id and to_id cols

        df_range_curr = df_range[df_range["to_id"] == df_cir.iloc[i]["to_id"]]
        df_range_curr = df_range_curr[
            df_range_curr["from_id"] == df_cir.iloc[i]["from_id"]
        ]
        range_idx = (df_range_curr["timestamp"] - timestamp).abs().idxmin()
        range = df_range_curr.loc[range_idx, "range"]
        X_data.append((range**2) * cir)
        y_data.append(is_nlos_miluv(df_cir.iloc[i]["to_id"]))


X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
